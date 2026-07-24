from latent_wam.manifest_compact import compact_manifest


def _sidecar(content_key: str):
    return {
        "filename": f"{content_key}.json",
        "file_count": 2,
        "variants": [
            {
                "content": {content_key: {"mean": [1.0], "std": [2.0]}},
                "files": 2,
                "sample_paths": [f"one/{content_key}.json"],
            }
        ],
        "invalid_files": [],
    }


def test_compact_manifest_retains_only_requested_source_sidecar_details():
    report = {
        "data_sources": [
            {"name": "oxe", "stats_gr00t_metadata": _sidecar("action")},
            {"name": "agibot_world", "stats_gr00t_metadata": _sidecar("state")},
        ]
    }
    compacted = compact_manifest(report, ("oxe",))
    assert compacted["data_sources"][0]["stats_gr00t_metadata"]["variants"][0][
        "content"
    ] == {"action": {"mean": [1.0], "std": [2.0]}}
    summary = compacted["data_sources"][1]["stats_gr00t_metadata"]
    assert summary == {
        "filename": "state.json",
        "file_count": 2,
        "variant_count": 1,
        "files_described_by_variants": 2,
        "sample_paths": ["one/state.json"],
        "invalid_files": [],
        "detail_omitted": True,
    }
    assert compacted["compaction"] == {
        "detailed_sidecar_sources": ["oxe"],
        "summarized_sidecar_sources": ["agibot_world"],
    }
