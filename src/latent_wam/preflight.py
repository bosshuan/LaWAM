from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from latent_wam.config import load_config, resolve_output_root
from latent_wam.data.intern_data_a1 import _feature_size, _select_feature_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an 8xA100 LatentWAM server")
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
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_info_files(root: Path, patterns, excluded) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    return [
        path
        for path in sorted(paths)
        if not any(token in str(path) for token in excluded)
    ]


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
        data = dataclasses.replace(data, root=args.data_root)
    config = dataclasses.replace(config, model=model, data=data)
    config.validate()
    checkpoint = Path(config.model.checkpoint).expanduser()
    data_root = Path(config.data.root).expanduser()
    output_root = resolve_output_root(config)
    failures = []
    report = {
        "config": dataclasses.asdict(config),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_file(),
        "data_root": str(data_root),
        "data_root_exists": data_root.is_dir(),
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
    if torch.cuda.device_count() != 8:
        failures.append(f"Expected exactly 8 visible GPUs, found {torch.cuda.device_count()}")
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
        non_a100 = [entry["name"] for entry in report["devices"] if "A100" not in entry["name"]]
        if non_a100:
            failures.append(f"Non-A100 devices are visible: {non_a100}")
    if not checkpoint.is_file():
        failures.append(f"Missing local checkpoint: {checkpoint}")
    elif not args.skip_checksum:
        report["checkpoint_sha256"] = sha256(checkpoint)
        if args.checkpoint_sha256 and report["checkpoint_sha256"] != args.checkpoint_sha256:
            failures.append("Checkpoint SHA256 does not match the requested digest")
    if not data_root.is_dir():
        failures.append(f"Missing InternData-A1 directory: {data_root}")
    else:
        info_files = discover_info_files(
            data_root, config.data.include_globs, config.data.exclude_contains
        )
        report["lerobot_v21_subdatasets"] = len(info_files)
        report["sample_info_files"] = [str(path) for path in info_files[:10]]
        if not info_files:
            failures.append("No candidate meta/info.json files were found")
        normalization_sources = {
            "stats.json": 0,
            "episodes_stats.jsonl": 0,
            "missing": 0,
        }
        missing_normalization = []
        for info_path in info_files:
            meta = info_path.parent
            if (meta / "stats.json").is_file():
                normalization_sources["stats.json"] += 1
            elif (meta / "episodes_stats.jsonl").is_file():
                normalization_sources["episodes_stats.jsonl"] += 1
            else:
                normalization_sources["missing"] += 1
                missing_normalization.append(str(meta))
        report["normalization_sources"] = normalization_sources
        report["sample_missing_normalization"] = missing_normalization[:10]
        if missing_normalization:
            failures.append(
                f"{len(missing_normalization)} subdatasets have neither stats.json "
                "nor episodes_stats.jsonl"
            )
        schema_variants: dict[str, dict] = {}
        unsupported_control_schema = []
        invalid_info_files = []
        for info_path in info_files:
            try:
                with info_path.open("r", encoding="utf-8") as handle:
                    info = json.load(handle)
                features = info.get("features", {})
                action_keys = _select_feature_keys(features, "actions.")
                state_keys = _select_feature_keys(features, "states.")
                if not action_keys or not state_keys:
                    unsupported_control_schema.append(str(info_path))
                    continue
                robot_type = str(info.get("robot_type", info_path.parent.parent.name))
                variant_name = f"{robot_type}:{'|'.join(action_keys)}"
                variant = schema_variants.setdefault(
                    variant_name,
                    {
                        "robot_type": robot_type,
                        "action_keys": list(action_keys),
                        "state_keys": list(state_keys),
                        "action_dim": sum(_feature_size(features[key]) for key in action_keys),
                        "state_dim": sum(_feature_size(features[key]) for key in state_keys),
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
        report["control_schema_variants"] = list(schema_variants.values())
        report["unsupported_control_schema_count"] = len(unsupported_control_schema)
        report["sample_unsupported_control_schema"] = unsupported_control_schema[:10]
        report["invalid_info_files"] = invalid_info_files[:10]
        if info_files and not schema_variants:
            failures.append("No subdataset exposes a supported joint/gripper control schema")
        if invalid_info_files:
            failures.append(f"{len(invalid_info_files)} meta/info.json files are invalid")
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
    serialized = json.dumps(report, indent=2)
    print(serialized, flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
