import json
import dataclasses
from pathlib import Path

import pytest

from latent_wam.config import ExperimentConfig
from latent_wam.preflight import (
    audit_data_source,
    audit_jsonl_first_records,
    json_default,
)


def test_preflight_report_serializes_transformers_sets():
    report = {
        "missing_keys": {"encoder.block.1", "encoder.block.0"},
        "path": Path("/tmp/t5-large"),
    }
    encoded = json.dumps(report, default=json_default)
    decoded = json.loads(encoded)
    assert decoded["missing_keys"] == ["encoder.block.0", "encoder.block.1"]
    assert decoded["path"] == "/tmp/t5-large"


def test_preflight_json_default_rejects_unknown_objects():
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_default(object())


def test_strict_manifest_reports_unknown_control_fields(tmp_path):
    meta = tmp_path / "dataset" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "features": {
                    "action": {"dtype": "float32", "shape": [7]},
                    "observation.state": {"dtype": "float32", "shape": [7]},
                    "observation.images.main": {"dtype": "video", "shape": [3, 480, 640]},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "stats.json").write_text("{}\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text('{"task_index": 0, "task": "test"}\n', encoding="utf-8")
    base = ExperimentConfig()
    config = dataclasses.replace(
        base,
        data=dataclasses.replace(base.data, strict_manifest=True),
    )
    report, failures = audit_data_source("unknown", tmp_path, config)
    assert report["unsupported_control_schema_count"] == 1
    assert "action" in report["sample_unsupported_control_schema"][0]["feature_keys"]
    assert report["control_manifest_variants"] == [
        {
            "feature_specs": {
                "action": {"dtype": "float32", "shape": [7]},
                "observation.state": {"dtype": "float32", "shape": [7]},
            },
            "subdatasets": 1,
            "sample_info_files": ["dataset/meta/info.json"],
        }
    ]
    assert report["normalization_key_variants"] == [
        {
            "source": "stats.json",
            "keys": [],
            "subdatasets": 1,
            "sample_info_files": ["dataset/meta/info.json"],
        }
    ]
    assert any("unsupported control manifest" in failure for failure in failures)


def test_preflight_accepts_audited_robomind_stats_and_vector_schema(tmp_path):
    meta = tmp_path / "robomind" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "features": {
                    "action": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": ["action"],
                    },
                    "actions": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": ["actions"],
                    },
                    "observation.state": {
                        "dtype": "float32",
                        "shape": [7],
                        "names": ["observation.state"],
                    },
                    "image_top": {"dtype": "video", "shape": [3, 480, 640]},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta / "tasks.jsonl").write_text(
        '{"task_index": 0, "task": "test"}\n',
        encoding="utf-8",
    )
    stats_gr00t = {
        "action": {"mean": [0.0] * 7, "std": [1.0] * 7},
        "observation.state": {"mean": [0.0] * 7, "std": [1.0] * 7},
    }
    (meta / "stats_gr00t.json").write_text(
        json.dumps(stats_gr00t),
        encoding="utf-8",
    )
    base = ExperimentConfig()
    config = dataclasses.replace(
        base,
        data=dataclasses.replace(
            base.data,
            roots=("/robomind",),
            source_names=("robomind",),
            mixture_weights=(1.0,),
            control_adapter_overrides={
                "robomind": "robomind_joint_vector",
            },
        ),
    )
    report, failures = audit_data_source("robomind", tmp_path, config)
    assert report["stats_gr00t_metadata"] == {
        "filename": "stats_gr00t.json",
        "file_count": 1,
        "variants": [
            {
                "content": stats_gr00t,
                "files": 1,
                "sample_paths": ["robomind/meta/stats_gr00t.json"],
            }
        ],
        "invalid_files": [],
    }
    assert report["normalization_sources"]["stats_gr00t.json"] == 1
    assert report["supported_control_schema_count"] == 1
    assert report["unsupported_control_schema_count"] == 0
    assert failures == []


def test_jsonl_sidecar_audit_captures_only_first_nonempty_record(tmp_path):
    meta = tmp_path / "dataset" / "meta"
    meta.mkdir(parents=True)
    first = {
        "episode_index": 0,
        "stats": {
            "action": {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "count": [10],
            }
        },
    }
    second = {"episode_index": 1, "stats": {"action": {"mean": [99.0]}}}
    (meta / "episodes_stats.jsonl").write_text(
        "\n" + json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    assert audit_jsonl_first_records(
        tmp_path,
        "episodes_stats.jsonl",
        (),
    ) == {
        "filename": "episodes_stats.jsonl",
        "file_count": 1,
        "variants": [
            {
                "first_record": first,
                "files": 1,
                "sample_paths": ["dataset/meta/episodes_stats.jsonl"],
                "sample_line_numbers": [2],
            }
        ],
        "invalid_files": [],
        "scope": "first non-empty record per file",
    }
