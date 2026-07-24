from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from latent_wam.config import load_config
from latent_wam.data.intern_data_a1 import InternDataA1Dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decode one real training sample from every configured data source"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-subdatasets", type=int, default=3)
    parser.add_argument("--max-episodes-per-subdataset", type=int, default=2)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only a one-line result instead of the JSON report",
    )
    return parser.parse_args()


def _tensor_summary(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()),
    }


def _validate_sample(sample, config) -> list[str]:
    failures: list[str] = []
    expected_shapes = {
        "context_rgb": (
            3,
            config.video.context_frames,
            config.video.resolution,
            config.video.resolution,
        ),
        "full_rgb": (
            3,
            config.video.context_frames + config.video.future_frames,
            config.video.resolution,
            config.video.resolution,
        ),
        "proprio": (
            config.action.proprio_history,
            config.action.max_proprio_dim,
        ),
        "proprio_valid": (
            config.action.proprio_history,
            config.action.max_proprio_dim,
        ),
        "past_actions": (
            config.action.past_action_history,
            config.action.max_action_dim,
        ),
        "past_action_valid": (config.action.past_action_history,),
        "actions": (
            config.action.chunk_size,
            config.action.max_action_dim,
        ),
        "action_valid": (
            config.action.chunk_size,
            config.action.max_action_dim,
        ),
        "gripper_mask": (
            config.action.chunk_size,
            config.action.max_action_dim,
        ),
    }
    shaped_tensors = {
        "context_rgb": sample.student.context_rgb,
        "full_rgb": sample.teacher.full_rgb,
        "proprio": sample.student.proprio,
        "proprio_valid": sample.student.proprio_valid,
        "past_actions": sample.student.past_actions,
        "past_action_valid": sample.student.past_action_valid,
        "actions": sample.targets.actions,
        "action_valid": sample.targets.action_valid,
        "gripper_mask": sample.targets.gripper_mask,
    }
    for name, value in shaped_tensors.items():
        if tuple(value.shape) != expected_shapes[name]:
            failures.append(
                f"{name} has shape {tuple(value.shape)}, expected {expected_shapes[name]}"
            )
    for name in ("context_rgb", "full_rgb", "proprio", "past_actions", "actions"):
        value = shaped_tensors[name]
        if not torch.isfinite(value).all():
            failures.append(f"{name} contains non-finite values")
    if not sample.student.instructions or not sample.student.instructions[0].strip():
        failures.append("sample has no non-empty language instruction")
    if not sample.targets.action_valid.any():
        failures.append("sample has no valid action components")
    if not sample.student.proprio_valid.any():
        failures.append("sample has no valid proprioception components")
    return failures


def probe_sources(config, max_subdatasets: int, max_episodes: int) -> dict:
    if max_subdatasets <= 0 or max_episodes <= 0:
        raise ValueError("probe dataset and episode limits must be positive")
    roots = config.data.roots or (config.data.root,)
    names = config.data.source_names or tuple(
        Path(root).expanduser().name for root in roots
    )
    reports: list[dict] = []
    failures: list[str] = []
    for root, name in zip(roots, names):
        source_data = dataclasses.replace(
            config.data,
            root=root,
            roots=(),
            source_names=(),
            mixture_weights=(),
            control_adapter_overrides={},
            mixture_epoch_samples=None,
            max_subdatasets=max_subdatasets,
            max_episodes_per_subdataset=max_episodes,
            fixed_sample_index=0,
        )
        source_config = dataclasses.replace(config, data=source_data)
        source_report: dict[str, object] = {
            "name": name,
            "root": root,
            "adapter_override": config.data.control_adapter_overrides.get(name),
        }
        try:
            dataset = InternDataA1Dataset(
                source_config,
                split="train",
                adapter_override=config.data.control_adapter_overrides.get(name),
            )
            sample = dataset[0]
            sample_failures = _validate_sample(sample, config)
            source_report.update(
                {
                    "passed": not sample_failures,
                    "subdatasets_discovered": len(dataset.subdatasets),
                    "episodes_discovered": len(dataset.episodes),
                    "context_rgb": _tensor_summary(sample.student.context_rgb),
                    "full_rgb": _tensor_summary(sample.teacher.full_rgb),
                    "proprio": _tensor_summary(sample.student.proprio),
                    "past_actions": _tensor_summary(sample.student.past_actions),
                    "actions": _tensor_summary(sample.targets.actions),
                    "instruction": sample.student.instructions[0],
                    "metadata": sample.targets.metadata,
                    "failures": sample_failures,
                }
            )
            failures.extend(f"{name}: {failure}" for failure in sample_failures)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            source_report.update({"passed": False, "failures": [message]})
            failures.append(f"{name}: {message}")
        reports.append(source_report)
    return {
        "audit_kind": "runtime_data_probe",
        "passed": not failures,
        "max_subdatasets_per_source": max_subdatasets,
        "max_episodes_per_subdataset": max_episodes,
        "sources": reports,
        "failures": failures,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    report = probe_sources(
        config,
        max_subdatasets=args.max_subdatasets,
        max_episodes=args.max_episodes_per_subdataset,
    )
    output = Path(args.output).expanduser().resolve()
    report["report_path"] = str(output)
    serialized = json.dumps(report, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized + "\n", encoding="utf-8")
    if args.quiet:
        print(
            f"Runtime data probe passed={report['passed']} output={output} "
            f"failures={len(report['failures'])}",
            flush=True,
        )
    else:
        print(serialized, flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
