from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from moe_infinity.utils.qwen_benchmark import qwen_benchmark_variant_status


DEFAULT_TRACES = ["mixed", "recurrence_heavy", "stationary"]
DEFAULT_VARIANTS = [
    "on_demand",
    "history_reuse_backbone",
    "history_reuse_consensus_backbone",
    "history_reuse_local_backbone",
]
LOCAL_VARIANT = "history_reuse_local_backbone"
COMPARISON_BASELINES = [
    "history_reuse_consensus_backbone",
    "history_reuse_backbone",
    "on_demand",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize pressure-sweep Qwen benchmark results without rerunning GPU "
            "experiments."
        )
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to <benchmark-root>/analysis.",
    )
    parser.add_argument("--traces", nargs="+", default=DEFAULT_TRACES)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--non-pressure-hit-threshold",
        type=float,
        default=0.99,
        help="Hit-rate threshold for labeling a completed case as non-pressure/control.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct_delta(new: float, old: float) -> float | None:
    if old == 0.0:
        return None
    return ((new - old) / old) * 100.0


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _discover_ratio_dirs(benchmark_root: Path) -> list[Path]:
    return sorted(
        path
        for path in benchmark_root.iterdir()
        if path.is_dir() and path.name.startswith("ratio_")
    )


def _case_status(raw_path: Path, log_path: Path) -> str:
    if raw_path.is_file():
        raw = _load_json(raw_path)
        aggregate = raw.get("aggregate", {})
        if int(aggregate.get("failure_count", 0)) > 0:
            return "failed"
        return "complete"
    if not log_path.is_file():
        return "missing"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failure_markers = ["Traceback", "DLOG_FATAL", "RuntimeError", "Aborted"]
    if any(marker in text for marker in failure_markers):
        return "failed"
    return "partial"


def _progress_counters_available(raw: Mapping[str, Any]) -> bool:
    progress_counter_keys = {
        "cache_hit_fetch_count",
        "cache_miss_fetch_count",
        "eviction_count",
        "all_locked_event_count",
        "no_victim_wait_count",
        "no_victim_wait_total_us",
        "no_victim_wait_max_us",
    }
    return any(
        bool(
            progress_counter_keys.intersection(
                record.get("dispatcher_stats", {}).keys()
            )
        )
        for record in raw.get("records", [])
        if isinstance(record, Mapping)
    )


def _benchmark_mode(raw: Mapping[str, Any]) -> str:
    return str(raw.get("benchmark_mode", "generate"))


def _performance_comparison_allowed(
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
) -> bool:
    return (
        lhs.get("benchmark_mode") == "forward"
        and rhs.get("benchmark_mode") == "forward"
        and lhs.get("benchmark_variant_status") == "stable"
        and rhs.get("benchmark_variant_status") == "stable"
        and lhs.get("pressure_label") != "progress-boundary"
        and rhs.get("pressure_label") != "progress-boundary"
    )


def _pressure_label(
    status: str,
    aggregate: Mapping[str, Any],
    *,
    hit_threshold: float,
) -> str:
    if status != "complete":
        return "unknown"
    hit_rate = _safe_float(aggregate.get("mean_cache_hit_rate"))
    busy_waits = _safe_float(aggregate.get("mean_dispatcher_busy_wait_count"))
    evictions = _safe_float(aggregate.get("mean_dispatcher_eviction_count"))
    all_locked = _safe_float(aggregate.get("mean_dispatcher_all_locked_event_count"))
    no_victim_waits = _safe_float(aggregate.get("mean_dispatcher_no_victim_wait_count"))
    if all_locked > 0.0 or no_victim_waits > 0.0:
        return "progress-boundary"
    if hit_rate >= hit_threshold and busy_waits == 0.0 and evictions == 0.0:
        return "non-pressure/control"
    return "pressure"


def _case_summary(
    *,
    ratio_dir: Path,
    trace_name: str,
    variant: str,
    hit_threshold: float,
) -> Dict[str, Any]:
    raw_path = ratio_dir / "raw" / f"{trace_name}__{variant}.json"
    log_path = ratio_dir / "logs" / f"{trace_name}__{variant}.log"
    status = _case_status(raw_path, log_path)
    raw: Dict[str, Any] = _load_json(raw_path) if raw_path.is_file() else {}
    aggregate = dict(raw.get("aggregate", {}))
    return {
        "ratio": ratio_dir.name,
        "trace_name": trace_name,
        "variant": variant,
        "status": status,
        "pressure_label": _pressure_label(
            status,
            aggregate,
            hit_threshold=hit_threshold,
        ),
        "raw_path": str(raw_path) if raw_path.is_file() else None,
        "log_path": str(log_path) if log_path.is_file() else None,
        "has_progress_counters": _progress_counters_available(raw),
        "benchmark_mode": _benchmark_mode(raw),
        "benchmark_variant_status": raw.get(
            "benchmark_variant_status",
            qwen_benchmark_variant_status(variant),
        ),
        "aggregate": aggregate,
        "fixed_new_tokens": raw.get("fixed_new_tokens"),
        "max_new_tokens": raw.get("max_new_tokens"),
        "request_count": aggregate.get("request_count"),
        "success_count": aggregate.get("success_count"),
        "failure_count": aggregate.get("failure_count"),
    }


def _index_cases(
    cases: Iterable[Mapping[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Mapping[str, Any]]]]:
    indexed: Dict[str, Dict[str, Dict[str, Mapping[str, Any]]]] = {}
    for case in cases:
        ratio = str(case["ratio"])
        trace = str(case["trace_name"])
        variant = str(case["variant"])
        indexed.setdefault(ratio, {}).setdefault(trace, {})[variant] = case
    return indexed


def _build_comparisons(cases: list[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    indexed = _index_cases(cases)
    comparisons: list[Dict[str, Any]] = []
    for ratio, traces in sorted(indexed.items()):
        for trace_name, variants in sorted(traces.items()):
            local = variants.get(LOCAL_VARIANT)
            if local is None or local.get("status") != "complete":
                continue
            local_agg = local.get("aggregate", {})
            for baseline_name in COMPARISON_BASELINES:
                baseline = variants.get(baseline_name)
                if baseline is None or baseline.get("status") != "complete":
                    continue
                baseline_agg = baseline.get("aggregate", {})
                performance_allowed = _performance_comparison_allowed(
                    local,
                    baseline,
                )
                comparisons.append(
                    {
                        "ratio": ratio,
                        "trace_name": trace_name,
                        "lhs_variant": LOCAL_VARIANT,
                        "rhs_variant": baseline_name,
                        "performance_comparison_allowed": performance_allowed,
                        "performance_guardrail": (
                            "ok"
                            if performance_allowed
                            else "performance deltas suppressed unless both cases are stable forward-mode non-boundary results"
                        ),
                        "tokens_per_second_delta_pct": (
                            _pct_delta(
                                _safe_float(
                                    local_agg.get("generated_tokens_per_second")
                                ),
                                _safe_float(
                                    baseline_agg.get("generated_tokens_per_second")
                                ),
                            )
                            if performance_allowed
                            else None
                        ),
                        "latency_per_token_delta_pct": (
                            _pct_delta(
                                _safe_float(
                                    local_agg.get(
                                        "latency_per_generated_token_mean_ms"
                                    )
                                ),
                                _safe_float(
                                    baseline_agg.get(
                                        "latency_per_generated_token_mean_ms"
                                    )
                                ),
                            )
                            if performance_allowed
                            else None
                        ),
                        "p95_latency_delta_pct": (
                            _pct_delta(
                                _safe_float(local_agg.get("latency_p95_s")),
                                _safe_float(baseline_agg.get("latency_p95_s")),
                            )
                            if performance_allowed
                            else None
                        ),
                    }
                )
    return comparisons


def _coverage_summary(
    *,
    ratio_name: str,
    cases: list[Mapping[str, Any]],
    expected_count: int,
) -> Dict[str, Any]:
    counts: Dict[str, int] = {
        "complete": 0,
        "partial": 0,
        "missing": 0,
        "failed": 0,
        "non_pressure": 0,
        "pressure": 0,
        "progress_boundary": 0,
        "unknown_pressure": 0,
    }
    for case in cases:
        status = str(case["status"])
        label = str(case["pressure_label"])
        counts[status] = counts.get(status, 0) + 1
        if label == "non-pressure/control":
            counts["non_pressure"] += 1
        elif label == "pressure":
            counts["pressure"] += 1
        elif label == "progress-boundary":
            counts["progress_boundary"] += 1
        else:
            counts["unknown_pressure"] += 1
    return {
        "ratio": ratio_name,
        "expected_cases": expected_count,
        "observed_cases": len(cases),
        **counts,
    }


def build_summary(
    *,
    benchmark_root: Path,
    traces: list[str],
    variants: list[str],
    hit_threshold: float,
) -> Dict[str, Any]:
    ratio_dirs = _discover_ratio_dirs(benchmark_root)
    expected_per_ratio = len(traces) * len(variants)
    cases: list[Dict[str, Any]] = []
    coverage: list[Dict[str, Any]] = []
    for ratio_dir in ratio_dirs:
        ratio_cases = [
            _case_summary(
                ratio_dir=ratio_dir,
                trace_name=trace_name,
                variant=variant,
                hit_threshold=hit_threshold,
            )
            for trace_name in traces
            for variant in variants
        ]
        cases.extend(ratio_cases)
        coverage.append(
            _coverage_summary(
                ratio_name=ratio_dir.name,
                cases=ratio_cases,
                expected_count=expected_per_ratio,
            )
        )
    return {
        "benchmark_root": str(benchmark_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "traces": traces,
        "variants": variants,
        "non_pressure_hit_threshold": hit_threshold,
        "coverage": coverage,
        "cases": cases,
        "comparisons": _build_comparisons(cases),
    }


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Pressure Sweep Summary",
        "",
        f"- Benchmark root: `{summary['benchmark_root']}`",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Non-pressure hit-rate threshold: `{summary['non_pressure_hit_threshold']}`",
        "",
        "## Coverage",
        "",
        "| Ratio | Complete | Partial | Failed | Missing | Non-pressure | Pressure | Progress-boundary | Unknown pressure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["coverage"]:
        lines.append(
            "| {ratio} | {complete} | {partial} | {failed} | {missing} | {non_pressure} | {pressure} | {progress_boundary} | {unknown_pressure} |".format(
                **item
            )
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Ratio | Trace | Variant | Mode | Variant status | Status | Pressure label | counters | tok/s | mean ms/tok | p95 s | hit rate | busy waits | evictions | all-locked | no-victim us |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for case in summary["cases"]:
        aggregate = case.get("aggregate", {})
        lines.append(
            "| {ratio} | {trace} | {variant} | {mode} | {variant_status} | {status} | {label} | {counters} | {tps} | {ms_tok} | {p95} | {hit} | {busy} | {evict} | {all_locked} | {no_victim_us} |".format(
                ratio=case["ratio"],
                trace=case["trace_name"],
                variant=case["variant"],
                mode=case.get("benchmark_mode", "generate"),
                variant_status=case.get("benchmark_variant_status", "experimental"),
                status=case["status"],
                label=case["pressure_label"],
                counters=str(case["has_progress_counters"]).lower(),
                tps=_fmt(aggregate.get("generated_tokens_per_second"), 3),
                ms_tok=_fmt(
                    aggregate.get("latency_per_generated_token_mean_ms"), 3
                ),
                p95=_fmt(aggregate.get("latency_p95_s"), 4),
                hit=_fmt(aggregate.get("mean_cache_hit_rate"), 4),
                busy=_fmt(aggregate.get("mean_dispatcher_busy_wait_count"), 2),
                evict=_fmt(aggregate.get("mean_dispatcher_eviction_count"), 2),
                all_locked=_fmt(
                    aggregate.get("mean_dispatcher_all_locked_event_count"), 2
                ),
                no_victim_us=_fmt(
                    aggregate.get("mean_dispatcher_no_victim_wait_total_us"), 1
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Local Comparisons",
            "",
            "| Ratio | Trace | Local vs | Perf allowed | tok/s delta % | mean ms/tok delta % | p95 delta % |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for comparison in summary["comparisons"]:
        lines.append(
            "| {ratio} | {trace} | {rhs} | {allowed} | {tps} | {ms_tok} | {p95} |".format(
                ratio=comparison["ratio"],
                trace=comparison["trace_name"],
                rhs=comparison["rhs_variant"],
                allowed=str(
                    comparison.get("performance_comparison_allowed", False)
                ).lower(),
                tps=_fmt(comparison.get("tokens_per_second_delta_pct"), 2),
                ms_tok=_fmt(comparison.get("latency_per_token_delta_pct"), 2),
                p95=_fmt(comparison.get("p95_latency_delta_pct"), 2),
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- `non-pressure/control` means the case completed with near-perfect cache hits and no observed dispatcher pressure; it should not be used as main paging-pressure evidence.",
            "- `progress-boundary` means the run observed all-locked/no-victim progress pressure and should be treated as robustness evidence, not a normal performance point.",
            "- `partial` and `missing` rows are included so an in-flight sweep can be inspected without waiting for every case to finish.",
            "- Local-vs-baseline performance deltas are suppressed unless both cases are stable forward-mode non-boundary results.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    if not benchmark_root.is_dir():
        raise ValueError(f"benchmark root does not exist: {benchmark_root}")
    output_root = Path(args.output_root) if args.output_root else benchmark_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    summary = build_summary(
        benchmark_root=benchmark_root,
        traces=list(args.traces),
        variants=list(args.variants),
        hit_threshold=float(args.non_pressure_hit_threshold),
    )
    (output_root / "pressure_sweep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "pressure_sweep_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
