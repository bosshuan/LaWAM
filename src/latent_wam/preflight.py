from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from latent_wam.config import load_config, resolve_output_root
from latent_wam.data.intern_data_a1 import (
    _feature_size,
    _load_norms,
    _select_control_feature_keys,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a LatentWAM training node")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--text-model")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument(
        "--verify-text-model-load",
        action="store_true",
        help="Load all local T5 encoder weights on CPU after checking its tokenizer/config",
    )
    parser.add_argument("--output")
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--expected-device-substring", default="A100")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_default(value):
    """Serialize container types returned by Transformers loading reports."""
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def discover_info_files(root: Path, patterns, excluded) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    return [
        path
        for path in sorted(paths)
        if not any(token in str(path) for token in excluded)
    ]


_NON_CONTROL_FEATURES = {
    "episode_index",
    "frame_index",
    "index",
    "language_raw",
    "task_index",
    "timestamp",
}


def compact_control_feature_specs(features: dict) -> dict[str, dict]:
    """Keep enough numeric manifest detail to design an adapter safely."""
    result: dict[str, dict] = {}
    for key in sorted(features):
        value = features[key]
        if not isinstance(value, dict):
            continue
        dtype = str(value.get("dtype", "unspecified"))
        if key in _NON_CONTROL_FEATURES or dtype in {"video", "string"}:
            continue
        result[key] = {
            field: value[field]
            for field in ("dtype", "shape", "names")
            if field in value
        }
    return result


def inspect_normalization_keys(meta: Path) -> tuple[str, list[str]]:
    """Read only the normalization key names, including one JSONL row at most."""
    stats_path = meta / "stats.json"
    if stats_path.is_file():
        with stats_path.open("r", encoding="utf-8") as handle:
            stats = json.load(handle)
        return "stats.json", sorted(stats) if isinstance(stats, dict) else []

    episode_stats_path = meta / "episodes_stats.jsonl"
    if episode_stats_path.is_file():
        with episode_stats_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                stats = row.get("stats", {}) if isinstance(row, dict) else {}
                return (
                    "episodes_stats.jsonl",
                    sorted(stats) if isinstance(stats, dict) else [],
                )
        return "episodes_stats.jsonl", []
    return "missing", []


def audit_json_sidecars(
    root: Path,
    filename: str,
    excluded: tuple[str, ...],
) -> dict:
    """Capture dataset-specific JSON metadata without assigning it semantics."""
    paths = [
        path
        for path in sorted(root.glob(f"**/{filename}"))
        if not any(token in str(path) for token in excluded)
    ]
    variants: dict[str, dict] = {}
    invalid_files: list[dict[str, str]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                content = json.load(handle)
            signature = json.dumps(content, sort_keys=True)
            variant = variants.setdefault(
                signature,
                {
                    "content": content,
                    "files": 0,
                    "sample_paths": [],
                },
            )
            variant["files"] += 1
            if len(variant["sample_paths"]) < 5:
                variant["sample_paths"].append(str(path.relative_to(root)))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            invalid_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "filename": filename,
        "file_count": len(paths),
        "variants": list(variants.values()),
        "invalid_files": invalid_files,
    }


def audit_local_t5(model_name: str, verify_weights: bool) -> tuple[dict, list[str]]:
    """Validate the exact offline T5 path used by training."""
    report = {
        "text_model": model_name,
        "local_t5_available": False,
        "t5_weights_verified": False,
    }
    failures: list[str] = []
    try:
        from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

        text_config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        report["t5_architecture"] = {
            "model_type": text_config.model_type,
            "d_model": getattr(text_config, "d_model", None),
            "d_ff": getattr(text_config, "d_ff", None),
            "num_layers": getattr(text_config, "num_layers", None),
            "num_heads": getattr(text_config, "num_heads", None),
            "vocab_size": getattr(text_config, "vocab_size", None),
        }
        if (
            text_config.model_type != "t5"
            or getattr(text_config, "d_model", None) != 1024
            or getattr(text_config, "num_layers", None) != 24
        ):
            failures.append(
                "The local text checkpoint is not T5-large "
                "(expected model_type=t5, d_model=1024, num_layers=24)"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        encoded = tokenizer(
            "move the robot arm",
            return_tensors="pt",
            truncation=True,
            max_length=16,
        )
        report["t5_tokenizer"] = {
            "class": type(tokenizer).__name__,
            "probe_tokens": int(encoded["input_ids"].shape[-1]),
        }
        report["local_t5_available"] = True

        if verify_weights:
            text_encoder, loading_info = T5EncoderModel.from_pretrained(
                model_name,
                local_files_only=True,
                output_loading_info=True,
            )
            missing_keys = loading_info.get("missing_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            error_messages = loading_info.get("error_msgs", [])
            report["t5_encoder_class"] = type(text_encoder).__name__
            report["t5_encoder_parameters"] = sum(
                parameter.numel() for parameter in text_encoder.parameters()
            )
            report["t5_weight_loading"] = {
                "missing_keys": missing_keys,
                "mismatched_keys": mismatched_keys,
                "error_messages": error_messages,
                "unexpected_key_count": len(
                    loading_info.get("unexpected_keys", [])
                ),
            }
            report["t5_weights_verified"] = not (
                missing_keys or mismatched_keys or error_messages
            )
            if not report["t5_weights_verified"]:
                failures.append(
                    "The local T5 encoder has missing, mismatched, or unreadable weights"
                )
            del text_encoder
    except Exception as error:
        report["t5_load_error"] = f"{type(error).__name__}: {error}"
        failures.append(
            "Failed to load the offline T5 tokenizer/config"
            + (" and encoder weights" if verify_weights else "")
        )
    return report, failures


def audit_data_source(name: str, root: Path, config) -> tuple[dict, list[str]]:
    """Audit one recursive LeRobot v2.1 source without decoding video."""
    failures: list[str] = []
    report: dict = {
        "name": name,
        "root": str(root),
        "root_exists": root.is_dir(),
    }
    if not root.is_dir():
        failures.append(f"{name}: missing data root {root}")
        return report, failures

    info_files = discover_info_files(
        root, config.data.include_globs, config.data.exclude_contains
    )
    report["lerobot_v21_subdatasets"] = len(info_files)
    report["sample_info_files"] = [str(path) for path in info_files[:10]]
    report["stats_gr00t_metadata"] = audit_json_sidecars(
        root,
        "stats_gr00t.json",
        config.data.exclude_contains,
    )
    if not info_files:
        failures.append(f"{name}: no candidate meta/info.json files were found")

    normalization_sources = {
        "stats.json": 0,
        "episodes_stats.jsonl": 0,
        "missing": 0,
    }
    missing_normalization = []
    schema_variants: dict[str, dict] = {}
    control_manifest_variants: dict[str, dict] = {}
    normalization_key_variants: dict[str, dict] = {}
    unsupported_control_schema = []
    invalid_info_files = []
    unsupported_versions = []
    missing_video_features = []
    missing_task_metadata = []
    oversized_control_schemas = []
    versions: dict[str, int] = {}
    robot_types: dict[str, int] = {}
    camera_keys: dict[str, int] = {}
    fps_values: dict[str, int] = {}
    licenses: dict[str, int] = {}
    for info_path in info_files:
        meta = info_path.parent
        if (meta / "stats.json").is_file():
            normalization_sources["stats.json"] += 1
        elif (meta / "episodes_stats.jsonl").is_file():
            normalization_sources["episodes_stats.jsonl"] += 1
        else:
            normalization_sources["missing"] += 1
            missing_normalization.append(str(meta))
        try:
            with info_path.open("r", encoding="utf-8") as handle:
                info = json.load(handle)
            version = str(info.get("codebase_version", "unspecified"))
            versions[version] = versions.get(version, 0) + 1
            if version != "unspecified" and not version.startswith("v2"):
                unsupported_versions.append(
                    {"path": str(info_path), "codebase_version": version}
                )
                continue
            features = info.get("features", {})
            relative_info_path = str(info_path.relative_to(root))
            control_feature_specs = compact_control_feature_specs(features)
            control_signature = json.dumps(control_feature_specs, sort_keys=True)
            control_variant = control_manifest_variants.setdefault(
                control_signature,
                {
                    "feature_specs": control_feature_specs,
                    "subdatasets": 0,
                    "sample_info_files": [],
                },
            )
            control_variant["subdatasets"] += 1
            if len(control_variant["sample_info_files"]) < 3:
                control_variant["sample_info_files"].append(relative_info_path)

            normalization_source, normalization_keys = inspect_normalization_keys(meta)
            normalization_signature = json.dumps(
                [normalization_source, normalization_keys], sort_keys=True
            )
            normalization_variant = normalization_key_variants.setdefault(
                normalization_signature,
                {
                    "source": normalization_source,
                    "keys": normalization_keys,
                    "subdatasets": 0,
                    "sample_info_files": [],
                },
            )
            normalization_variant["subdatasets"] += 1
            if len(normalization_variant["sample_info_files"]) < 3:
                normalization_variant["sample_info_files"].append(relative_info_path)
            fps = str(info.get("fps", "unspecified"))
            fps_values[fps] = fps_values.get(fps, 0) + 1
            license_name = str(info.get("license", "unspecified"))
            licenses[license_name] = licenses.get(license_name, 0) + 1
            action_keys, state_keys, adapter_name = _select_control_feature_keys(
                features
            )
            videos = sorted(
                key
                for key, value in features.items()
                if isinstance(value, dict) and value.get("dtype") == "video"
            )
            for camera in videos:
                camera_keys[camera] = camera_keys.get(camera, 0) + 1
            if not videos:
                missing_video_features.append(str(info_path))
            if not (meta / "tasks.jsonl").is_file():
                missing_task_metadata.append(str(meta))
            if not action_keys or not state_keys or adapter_name is None:
                unsupported_control_schema.append(
                    {
                        "path": str(info_path),
                        "feature_keys": sorted(features),
                        "video_keys": videos,
                    }
                )
                continue
            action_dim = sum(_feature_size(features[key]) for key in action_keys)
            state_dim = sum(_feature_size(features[key]) for key in state_keys)
            if (
                action_dim > config.action.max_action_dim
                or state_dim > config.action.max_proprio_dim
            ):
                oversized_control_schemas.append(
                    {
                        "path": str(info_path),
                        "action_dim": action_dim,
                        "state_dim": state_dim,
                    }
                )
            all_control_keys = (*action_keys, *state_keys)
            norms, normalization_path = _load_norms(
                info_path.parent.parent,
                all_control_keys,
            )
            missing_norm_keys = [
                key
                for key in all_control_keys
                if not ("gripper" in key and "openness" in key)
                and key not in norms
            ]
            if missing_norm_keys:
                raise ValueError(
                    f"{normalization_path} has no mean/std for {missing_norm_keys}"
                )
            for key, norm in norms.items():
                expected_size = _feature_size(features[key])
                if norm.mean.size != expected_size or norm.std.size != expected_size:
                    raise ValueError(
                        f"{normalization_path} statistics for {key} have shape "
                        f"({norm.mean.size}, {norm.std.size}), expected {expected_size}"
                    )
            robot_type = str(info.get("robot_type", info_path.parent.parent.name))
            robot_types[robot_type] = robot_types.get(robot_type, 0) + 1
            variant_name = json.dumps(
                {
                    "adapter": adapter_name,
                    "robot_type": robot_type,
                    "actions": [
                        [key, _feature_size(features[key])] for key in action_keys
                    ],
                    "states": [
                        [key, _feature_size(features[key])] for key in state_keys
                    ],
                },
                sort_keys=True,
            )
            variant = schema_variants.setdefault(
                variant_name,
                {
                    "adapter": adapter_name,
                    "robot_type": robot_type,
                    "action_keys": list(action_keys),
                    "state_keys": list(state_keys),
                    "action_dim": action_dim,
                    "state_dim": state_dim,
                    "binary_gripper": any(
                        "gripper" in key and "openness" in key
                        for key in action_keys
                    ),
                    "subdatasets": 0,
                },
            )
            variant["subdatasets"] += 1
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
            invalid_info_files.append({"path": str(info_path), "error": str(error)})

    report.update(
        {
            "codebase_versions": versions,
            "normalization_sources": normalization_sources,
            "sample_missing_normalization": missing_normalization[:10],
            "control_schema_variants": list(schema_variants.values()),
            "control_manifest_variants": list(control_manifest_variants.values()),
            "normalization_key_variants": list(normalization_key_variants.values()),
            "supported_control_schema_count": sum(
                variant["subdatasets"] for variant in schema_variants.values()
            ),
            "unsupported_control_schema_count": len(unsupported_control_schema),
            "sample_unsupported_control_schema": unsupported_control_schema[:10],
            "invalid_info_files": invalid_info_files[:10],
            "unsupported_version_count": len(unsupported_versions),
            "sample_unsupported_versions": unsupported_versions[:10],
            "missing_video_feature_count": len(missing_video_features),
            "sample_missing_video_features": missing_video_features[:10],
            "missing_task_metadata_count": len(missing_task_metadata),
            "sample_missing_task_metadata": missing_task_metadata[:10],
            "oversized_control_schema_count": len(oversized_control_schemas),
            "sample_oversized_control_schemas": oversized_control_schemas[:10],
            "robot_types": robot_types,
            "camera_keys": camera_keys,
            "fps_values": fps_values,
            "licenses": licenses,
        }
    )
    if missing_normalization:
        failures.append(
            f"{name}: {len(missing_normalization)} subdatasets have neither "
            "stats.json nor episodes_stats.jsonl"
        )
    if info_files and not schema_variants:
        failures.append(f"{name}: no supported joint/gripper control schema")
    if config.data.strict_manifest and unsupported_control_schema:
        failures.append(
            f"{name}: {len(unsupported_control_schema)} subdatasets use an "
            "unsupported control manifest"
        )
    if config.data.strict_manifest and unsupported_versions:
        failures.append(
            f"{name}: {len(unsupported_versions)} subdatasets are not LeRobot v2.x"
        )
    if config.data.strict_manifest and missing_video_features:
        failures.append(
            f"{name}: {len(missing_video_features)} subdatasets have no video feature"
        )
    if config.data.strict_manifest and missing_task_metadata:
        failures.append(
            f"{name}: {len(missing_task_metadata)} subdatasets have no tasks.jsonl"
        )
    if oversized_control_schemas:
        failures.append(
            f"{name}: {len(oversized_control_schemas)} subdatasets exceed padded "
            "action/state dimensions"
        )
    if invalid_info_files:
        failures.append(f"{name}: {len(invalid_info_files)} meta/info.json files are invalid")
    return report, failures


def main():
    args = parse_args()
    config = load_config(args.config)
    model = config.model
    data = config.data
    if args.checkpoint:
        model = dataclasses.replace(model, checkpoint=args.checkpoint)
    if args.text_model:
        model = dataclasses.replace(model, text_model=args.text_model)
    if args.data_root:
        data = dataclasses.replace(
            data,
            root=args.data_root,
            roots=(),
            source_names=(),
            mixture_weights=(),
            mixture_epoch_samples=None,
        )
    config = dataclasses.replace(config, model=model, data=data)
    config.validate()
    checkpoint = Path(config.model.checkpoint).expanduser()
    output_root = resolve_output_root(config)
    failures = []
    report = {
        "config": dataclasses.asdict(config),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "output_root": str(output_root),
    }
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    version_parts = torch.__version__.split("+")[0].split(".")[:2]
    try:
        torch_version = tuple(int(value) for value in version_parts)
    except ValueError:
        torch_version = (0, 0)
    report["torch_version"] = torch.__version__
    if torch_version < (2, 4):
        failures.append(f"PyTorch >= 2.4 is required, found {torch.__version__}")
    if args.expected_gpus <= 0:
        raise ValueError("--expected-gpus must be positive")
    if torch.cuda.device_count() != args.expected_gpus:
        failures.append(
            f"Expected exactly {args.expected_gpus} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        failures.append("Visible GPUs do not report bf16 support")
    if torch.cuda.is_available():
        report["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
        unexpected_devices = [
            entry["name"]
            for entry in report["devices"]
            if args.expected_device_substring.lower() not in entry["name"].lower()
        ]
        if unexpected_devices:
            failures.append(
                f"Devices do not match {args.expected_device_substring!r}: "
                f"{unexpected_devices}"
            )
    if not checkpoint.is_file():
        failures.append(f"Missing local checkpoint: {checkpoint}")
    elif not args.skip_checksum:
        report["checkpoint_sha256"] = sha256(checkpoint)
        if args.checkpoint_sha256 and report["checkpoint_sha256"] != args.checkpoint_sha256:
            failures.append("Checkpoint SHA256 does not match the requested digest")
    source_roots = config.data.roots or (config.data.root,)
    source_names = config.data.source_names or tuple(
        Path(root).expanduser().name for root in source_roots
    )
    source_weights = config.data.mixture_weights or (1.0,) * len(source_roots)
    weight_sum = sum(source_weights)
    report["data_mixture"] = {
        "strict_manifest": config.data.strict_manifest,
        "mixture_epoch_samples": config.data.mixture_epoch_samples,
        "weights": {
            name: weight / weight_sum
            for name, weight in zip(source_names, source_weights)
        },
    }
    data_reports = []
    for name, root_value in zip(source_names, source_roots):
        source_report, source_failures = audit_data_source(
            name,
            Path(root_value).expanduser(),
            config,
        )
        data_reports.append(source_report)
        failures.extend(source_failures)
    report["data_sources"] = data_reports
    if len(data_reports) == 1:
        source_report = data_reports[0]
        report["data_root"] = source_report["root"]
        report["data_root_exists"] = source_report["root_exists"]
        for key in (
            "lerobot_v21_subdatasets",
            "sample_info_files",
            "normalization_sources",
            "sample_missing_normalization",
            "control_schema_variants",
            "unsupported_control_schema_count",
            "sample_unsupported_control_schema",
            "invalid_info_files",
        ):
            report[key] = source_report.get(key)
    if config.model.text_backend == "t5":
        text_report, text_failures = audit_local_t5(
            config.model.text_model,
            verify_weights=args.verify_text_model_load,
        )
        report.update(text_report)
        failures.extend(text_failures)
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / ".write-probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as error:
        failures.append(f"Output directory is not writable: {error}")
    report["failures"] = failures
    output = Path(args.output).expanduser().resolve() if args.output else None
    if output is not None:
        report["report_path"] = str(output)
    serialized = json.dumps(report, indent=2, default=json_default)
    print(serialized, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
