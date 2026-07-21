from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare uninterrupted and resumed LatentWAM checkpoints"
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output")
    return parser.parse_args()


def _load_student(path: str | Path) -> tuple[int, dict[str, torch.Tensor]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    step = int(payload["step"])
    state = payload["student"]
    del payload
    return step, state


def main():
    args = parse_args()
    if args.atol < 0:
        raise ValueError("--atol cannot be negative")
    reference_step, reference = _load_student(args.reference)
    candidate_step, candidate = _load_student(args.candidate)
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    mismatched: list[dict[str, object]] = []
    max_abs_difference = 0.0
    for key in sorted(set(reference) & set(candidate)):
        expected = reference[key]
        actual = candidate[key]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            mismatched.append(
                {
                    "key": key,
                    "reference_shape": list(expected.shape),
                    "candidate_shape": list(actual.shape),
                    "reference_dtype": str(expected.dtype),
                    "candidate_dtype": str(actual.dtype),
                }
            )
            continue
        if torch.equal(expected, actual):
            continue
        if expected.is_floating_point() or expected.is_complex():
            difference = float((expected - actual).abs().max())
            max_abs_difference = max(max_abs_difference, difference)
            if difference <= args.atol:
                continue
        else:
            difference = None
        mismatched.append({"key": key, "max_abs_difference": difference})

    report = {
        "reference": str(Path(args.reference).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "reference_step": reference_step,
        "candidate_step": candidate_step,
        "atol": args.atol,
        "max_abs_difference": max_abs_difference,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_count": len(mismatched),
        "sample_mismatches": mismatched[:20],
    }
    serialized = json.dumps(report, indent=2)
    print(serialized, flush=True)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    if (
        reference_step != candidate_step
        or missing
        or unexpected
        or mismatched
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
