from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SIDECAR_FIELDS = (
    "stats_gr00t_metadata",
    "stats_delta_state_metadata",
    "episodes_stats_metadata",
)
DEFAULT_DETAIL_SOURCES = ("oxe", "robomind")


def _compact_sidecar(metadata: dict[str, Any]) -> dict[str, Any]:
    variants = metadata.get("variants", [])
    sample_paths: list[str] = []
    for variant in variants:
        for path in variant.get("sample_paths", []):
            if path not in sample_paths:
                sample_paths.append(path)
            if len(sample_paths) >= 10:
                break
        if len(sample_paths) >= 10:
            break
    result = {
        "filename": metadata.get("filename"),
        "file_count": metadata.get("file_count", 0),
        "variant_count": len(variants),
        "files_described_by_variants": sum(
            int(variant.get("files", 0)) for variant in variants
        ),
        "sample_paths": sample_paths,
        "invalid_files": metadata.get("invalid_files", []),
        "detail_omitted": True,
    }
    if "scope" in metadata:
        result["scope"] = metadata["scope"]
    return result


def compact_manifest(
    report: dict[str, Any],
    detail_sources: tuple[str, ...] = DEFAULT_DETAIL_SOURCES,
) -> dict[str, Any]:
    """Drop large sidecar arrays outside explicitly retained data sources.

    The input report is mutated in place to avoid doubling memory while
    compacting large storage manifests.
    """
    detail_source_set = set(detail_sources)
    for source in report.get("data_sources", []):
        if source.get("name") in detail_source_set:
            continue
        for field in SIDECAR_FIELDS:
            metadata = source.get(field)
            if isinstance(metadata, dict):
                source[field] = _compact_sidecar(metadata)
    report["compaction"] = {
        "detailed_sidecar_sources": list(detail_sources),
        "summarized_sidecar_sources": [
            source.get("name")
            for source in report.get("data_sources", [])
            if source.get("name") not in detail_source_set
        ],
    }
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a GitHub-sized LatentWAM storage manifest"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--detail-source",
        action="append",
        dest="detail_sources",
        help="Source whose complete sidecar statistics must be retained",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("compact manifest output must differ from its input")
    with input_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    detail_sources = tuple(args.detail_sources or DEFAULT_DETAIL_SOURCES)
    compact_manifest(report, detail_sources)
    report["compaction"]["detailed_report"] = str(input_path)
    report["report_path"] = str(output_path)
    serialized = json.dumps(report, indent=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(f"Compact storage manifest: {output_path}")
    print(f"Compact bytes: {output_path.stat().st_size}")


if __name__ == "__main__":
    main()
