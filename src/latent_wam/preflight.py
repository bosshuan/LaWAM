from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import torch

from latent_wam.config import load_config, resolve_output_root


def parse_args():
    parser = argparse.ArgumentParser(description="Validate an 8xA100 LatentWAM server")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-root")
    parser.add_argument("--text-model")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--skip-checksum", action="store_true")
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
    if config.model.text_backend == "t5":
        text_path = Path(config.model.text_model).expanduser()
        text_available = text_path.is_dir()
        if not text_available:
            try:
                from transformers.utils.hub import cached_file

                text_available = cached_file(
                    config.model.text_model,
                    "config.json",
                    local_files_only=True,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                ) is not None
            except Exception:
                text_available = False
        report["local_t5_available"] = text_available
        if not text_available:
            failures.append(
                "Frozen T5 is not available locally; pass --text-model to training "
                "or change model.text_model to an existing server directory"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    probe = output_root / ".write-probe"
    try:
        probe.touch()
        probe.unlink()
    except OSError as error:
        failures.append(f"Output directory is not writable: {error}")
    report["failures"] = failures
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
