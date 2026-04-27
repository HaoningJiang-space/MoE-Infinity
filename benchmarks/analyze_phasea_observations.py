from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from moe_infinity.analysis.phasea import (
    analyze_phasea_event_file,
    render_phasea_markdown_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Phase-A observation event logs from benchmark_qwen_offloading."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--reference-benchmark-root",
        default=None,
        help="Optional benchmark root without phasea instrumentation for overhead comparison.",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument(
        "--exclude-warmup-events",
        action="store_true",
        help="Ignore policy events emitted for warmup requests.",
    )
    return parser.parse_args()


def _load_raw_case(raw_path: Path) -> Dict[str, Any]:
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _merge_reference_overhead(
    analysis: Dict[str, Any],
    *,
    benchmark_root: Path,
    reference_root: Path | None,
) -> None:
    raw_name = Path(analysis["event_path"]).with_suffix(".json").name
    current_raw = benchmark_root / "raw" / raw_name
    if not current_raw.is_file():
        return
    current_payload = _load_raw_case(current_raw)
    current_latency = float(
        current_payload.get("aggregate", {}).get(
            "latency_per_generated_token_mean_ms", 0.0
        )
    )
    analysis["instrumentation_validation"]["observed_latency_per_token_ms"] = (
        current_latency
    )

    if reference_root is None:
        return
    reference_raw = reference_root / "raw" / raw_name
    if not reference_raw.is_file():
        return
    reference_payload = _load_raw_case(reference_raw)
    reference_latency = float(
        reference_payload.get("aggregate", {}).get(
            "latency_per_generated_token_mean_ms", 0.0
        )
    )
    if reference_latency > 0:
        overhead_ratio = (current_latency - reference_latency) / reference_latency
    else:
        overhead_ratio = 0.0
    analysis["instrumentation_validation"]["reference_latency_per_token_ms"] = (
        reference_latency
    )
    analysis["instrumentation_validation"]["latency_overhead_ratio"] = (
        overhead_ratio
    )


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    reference_root = (
        Path(args.reference_benchmark_root)
        if args.reference_benchmark_root
        else None
    )
    events_dir = benchmark_root / "events"
    if not events_dir.is_dir():
        raise ValueError(
            f"{benchmark_root} does not contain an events/ directory. "
            "Run benchmark_qwen_offloading.py with --phasea-events first."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    analyses: List[Dict[str, Any]] = []
    for path in sorted(events_dir.glob("*.jsonl")):
        analysis = analyze_phasea_event_file(
            path,
            horizons=args.horizons,
            budgets=args.budgets,
            exclude_warmup_events=args.exclude_warmup_events,
        )
        _merge_reference_overhead(
            analysis,
            benchmark_root=benchmark_root,
            reference_root=reference_root,
        )
        analyses.append(analysis)

    summary = {
        "benchmark_root": str(benchmark_root),
        "horizons": list(args.horizons),
        "budgets": list(args.budgets),
        "exclude_warmup_events": bool(args.exclude_warmup_events),
        "analyses": analyses,
    }
    (output_root / "phasea_observations.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "phasea_observations.md").write_text(
        render_phasea_markdown_summary(analyses=analyses),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
