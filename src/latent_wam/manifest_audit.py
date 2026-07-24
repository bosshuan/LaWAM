from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from latent_wam.config import load_config
from latent_wam.preflight import audit_data_source, json_default


DEFAULT_TRAIN_PREFIX = Path("/opt/huawei")
DEFAULT_STORAGE_PREFIX = Path("/home/ma-user/work")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Audit configured LeRobot manifests from a CPU storage view; "
            "this does not load CUDA, V-JEPA, or T5 weights"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--path-map-from",
        default=str(DEFAULT_TRAIN_PREFIX),
        help="Training-time path prefix present in the YAML config",
    )
    parser.add_argument(
        "--path-map-to",
        default=str(DEFAULT_STORAGE_PREFIX),
        help="Storage-time prefix available to the CPU audit process",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def remap_path(
    value: str | Path,
    source_prefix: str | Path,
    target_prefix: str | Path,
) -> Path:
    """Replace one exact leading path component sequence without touching suffixes."""
    path = Path(value).expanduser()
    source = Path(source_prefix).expanduser()
    target = Path(target_prefix).expanduser()
    try:
        suffix = path.relative_to(source)
    except ValueError as error:
        raise ValueError(f"{path} is not below configured prefix {source}") from error
    return target / suffix


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    config = load_config(config_path)

    configured_roots = config.data.roots or (config.data.root,)
    source_names = config.data.source_names or tuple(
        Path(root).expanduser().name for root in configured_roots
    )
    source_weights = config.data.mixture_weights or (1.0,) * len(configured_roots)
    weight_sum = sum(source_weights)

    failures: list[str] = []
    data_reports: list[dict] = []
    mapped_roots: list[Path] = []
    for name, configured_root in zip(source_names, configured_roots):
        try:
            storage_root = remap_path(
                configured_root,
                args.path_map_from,
                args.path_map_to,
            )
        except ValueError as error:
            failures.append(f"{name}: {error}")
            data_reports.append(
                {
                    "name": name,
                    "configured_training_root": configured_root,
                    "path_mapping_error": str(error),
                }
            )
            continue
        mapped_roots.append(storage_root)
        source_report, source_failures = audit_data_source(name, storage_root, config)
        source_report["configured_training_root"] = configured_root
        source_report["storage_audit_root"] = str(storage_root)
        if storage_root.is_dir():
            try:
                source_report["sample_top_level_entries"] = [
                    {
                        "name": path.name,
                        "kind": "directory" if path.is_dir() else "file",
                    }
                    for path in sorted(storage_root.iterdir(), key=lambda item: item.name)[:50]
                ]
            except OSError as error:
                source_report["top_level_listing_error"] = str(error)
                source_failures.append(f"{name}: could not list {storage_root}: {error}")
        data_reports.append(source_report)
        failures.extend(source_failures)

    report = {
        "audit_kind": "cpu_storage_manifest",
        "config_path": str(config_path),
        "config": dataclasses.asdict(config),
        "path_mapping": {
            "training_prefix": str(Path(args.path_map_from).expanduser()),
            "storage_prefix": str(Path(args.path_map_to).expanduser()),
        },
        "model_weights_loaded": False,
        "cuda_operations_requested": False,
        "data_mixture": {
            "strict_manifest": config.data.strict_manifest,
            "mixture_epoch_samples": config.data.mixture_epoch_samples,
            "weights": {
                name: weight / weight_sum
                for name, weight in zip(source_names, source_weights)
            },
        },
        "mapped_data_roots": [str(root) for root in mapped_roots],
        "data_sources": data_reports,
        "failures": failures,
        "passed": not failures,
        "report_path": str(output),
    }
    serialized = json.dumps(report, indent=2, default=json_default)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(f"Detailed storage manifest: {output}", flush=True)
    print(f"Passed: {report['passed']}", flush=True)
    print(f"Failure count: {len(failures)}", flush=True)
    for failure in failures:
        print(f"- {failure}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
