from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping


MODES = {
    "conservative": {"prefetch_future_layers": 2, "prefetch_max_candidates": 16},
    "aggressive": {"prefetch_future_layers": 4, "prefetch_max_candidates": 32},
}
TRACE_NAME = "mixed"
VARIANTS = [
    "on_demand",
    "history_reuse_consensus_backbone",
    "history_reuse_local_backbone",
]
FAILURE_MARKERS = [
    "evict_node is nullptr",
    "[FATAL",
    "DLOG_FATAL",
    "Traceback",
    "RuntimeError",
    "Aborted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize GPU0 strong-pressure canary logs and raw metrics."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to <benchmark-root>/analysis.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_int_counter(
    aggregate: Dict[str, Any],
    *,
    target_key: str,
    source: Mapping[str, Any],
    source_key: str,
) -> None:
    aggregate[target_key] = int(aggregate.get(target_key, 0) or 0) + int(
        source.get(source_key, 0) or 0
    )


def _merge_failure_snapshot(
    aggregate: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(aggregate)
    failure = raw.get("failure")
    if not isinstance(failure, Mapping):
        return merged
    if bool(failure.get("is_warmup", False)):
        return merged

    dispatcher = failure.get("dispatcher_stats", {})
    if isinstance(dispatcher, Mapping):
        dispatcher_counters = {
            "dispatcher_cache_hit_fetch_count_total": "cache_hit_fetch_count",
            "dispatcher_cache_miss_fetch_count_total": "cache_miss_fetch_count",
            "dispatcher_eviction_count_total": "eviction_count",
            "dispatcher_all_locked_event_count_total": "all_locked_event_count",
            "dispatcher_no_victim_wait_count_total": "no_victim_wait_count",
            "dispatcher_no_victim_wait_total_us": "no_victim_wait_total_us",
            "dispatcher_fetch_dequeue_count_total": "fetch_dequeue_count",
            "dispatcher_exec_dequeue_count_total": "exec_dequeue_count",
            "dispatcher_output_count_total": "output_count",
            "dispatcher_pending_wait_count_total": "pending_wait_count",
            "dispatcher_pending_wait_total_us": "pending_wait_total_us",
            "dispatcher_pending_stall_count_total": "pending_stall_count",
        }
        for target_key, source_key in dispatcher_counters.items():
            _add_int_counter(
                merged,
                target_key=target_key,
                source=dispatcher,
                source_key=source_key,
            )

    prefetcher = failure.get("prefetcher_stats", {})
    if isinstance(prefetcher, Mapping):
        prefetch_counters = {
            "prefetch_candidate_count_total": "prefetch_candidate_count",
            "prefetch_admitted_count_total": "prefetch_admitted_count",
            "prefetch_enqueue_count_total": "prefetch_enqueue_count",
            "prefetch_drop_count_total": "prefetch_drop_count",
            "prefetch_drop_cap_count_total": "prefetch_drop_cap_count",
            "prefetch_drop_pressure_count_total": "prefetch_drop_pressure_count",
            "prefetch_drop_no_evictable_count_total": (
                "prefetch_drop_no_evictable_count"
            ),
            "prefetch_under_pressure_count_total": "prefetch_under_pressure_count",
            "demand_prefetch_conflict_count_total": (
                "demand_prefetch_conflict_count"
            ),
        }
        for target_key, source_key in prefetch_counters.items():
            _add_int_counter(
                merged,
                target_key=target_key,
                source=prefetcher,
                source_key=source_key,
            )
        merged["pressure_locked_max"] = max(
            int(merged.get("pressure_locked_max", 0) or 0),
            int(prefetcher.get("pressure_locked_max", 0) or 0),
        )
        if int(prefetcher.get("pressure_sample_count", 0) or 0) > 0:
            failure_evictable = int(prefetcher.get("pressure_evictable_min", 0) or 0)
            current = merged.get("pressure_evictable_min")
            if current in {None, "n/a"}:
                merged["pressure_evictable_min"] = failure_evictable
            else:
                merged["pressure_evictable_min"] = min(
                    int(current or 0),
                    failure_evictable,
                )

    candidate_total = int(merged.get("prefetch_candidate_count_total", 0) or 0)
    if candidate_total > 0:
        merged["prefetch_admit_rate"] = float(
            int(merged.get("prefetch_admitted_count_total", 0) or 0)
            / candidate_total
        )
        merged["prefetch_pressure_drop_rate"] = float(
            int(merged.get("prefetch_drop_pressure_count_total", 0) or 0)
            / candidate_total
        )

    merged["failure_snapshot_merged"] = True
    return merged


def _load_status(benchmark_root: Path) -> Dict[tuple[str, str], Dict[str, str]]:
    status_path = benchmark_root / "case_status.tsv"
    if not status_path.is_file():
        return {}
    rows: Dict[tuple[str, str], Dict[str, str]] = {}
    lines = status_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return rows
    headers = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(headers):
            continue
        row = dict(zip(headers, parts))
        rows[(row.get("mode", ""), row.get("variant", ""))] = row
    return rows


def _log_counts(log_path: Path) -> Dict[str, Any]:
    if not log_path.is_file():
        return {
            "log_exists": False,
            "all_locked_count": 0,
            "progress_stall_count": 0,
            "fatal_count": 0,
            "failure_marker_count": 0,
            "first_all_locked": None,
            "last_all_locked": None,
        }
    text = log_path.read_text(encoding="utf-8", errors="replace")
    all_locked_lines = [
        line for line in text.splitlines() if "All cached expert locked" in line
    ]
    return {
        "log_exists": True,
        "all_locked_count": len(all_locked_lines),
        "progress_stall_count": text.count("progress stall"),
        "fatal_count": text.count("evict_node is nullptr"),
        "failure_marker_count": sum(text.count(marker) for marker in FAILURE_MARKERS),
        "first_all_locked": all_locked_lines[0] if all_locked_lines else None,
        "last_all_locked": all_locked_lines[-1] if all_locked_lines else None,
    }


def _case_status(raw_path: Path, log_info: Mapping[str, Any], exit_code: str | None) -> str:
    if raw_path.is_file() and exit_code in {None, "0"}:
        return "complete"
    if exit_code not in {None, "0"}:
        return "failed"
    if int(log_info.get("failure_marker_count", 0)) > 0:
        return "failed"
    if log_info.get("log_exists"):
        return "partial"
    return "missing"


def _case_summary(
    *,
    benchmark_root: Path,
    mode: str,
    variant: str,
    status_rows: Mapping[tuple[str, str], Mapping[str, str]],
) -> Dict[str, Any]:
    mode_root = benchmark_root / mode
    raw_path = mode_root / "raw" / f"{TRACE_NAME}__{variant}.json"
    log_path = mode_root / "logs" / f"{TRACE_NAME}__{variant}.log"
    status_row = dict(status_rows.get((mode, variant), {}))
    log_info = _log_counts(log_path)
    raw = _load_json(raw_path) if raw_path.is_file() else {}
    aggregate = _merge_failure_snapshot(dict(raw.get("aggregate", {})), raw)
    exit_code = status_row.get("exit_code")
    return {
        "mode": mode,
        "trace_name": TRACE_NAME,
        "variant": variant,
        "status": _case_status(raw_path, log_info, exit_code),
        "exit_code": exit_code,
        "started_utc": status_row.get("started_utc"),
        "finished_utc": status_row.get("finished_utc"),
        "raw_path": str(raw_path) if raw_path.is_file() else None,
        "log_path": str(log_path) if log_path.is_file() else None,
        "prefetch_future_layers": MODES[mode]["prefetch_future_layers"],
        "prefetch_max_candidates": MODES[mode]["prefetch_max_candidates"],
        "log_characterization": log_info,
        "aggregate": aggregate,
    }


def build_summary(benchmark_root: Path) -> Dict[str, Any]:
    status_rows = _load_status(benchmark_root)
    case_specs = list(status_rows.keys()) or [
        (mode, variant) for mode in MODES for variant in VARIANTS
    ]
    cases = [
        _case_summary(
            benchmark_root=benchmark_root,
            mode=mode,
            variant=variant,
            status_rows=status_rows,
        )
        for mode, variant in case_specs
    ]
    return {
        "benchmark_root": str(benchmark_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "trace_name": TRACE_NAME,
        "modes": MODES,
        "variants": VARIANTS,
        "cases": cases,
    }


def _render_markdown(summary: Mapping[str, Any]) -> str:
    any_on_demand_failed = any(
        case["variant"] == "on_demand" and case["status"] == "failed"
        for case in summary["cases"]
    )
    any_aggressive_failed = any(
        case["mode"] == "aggressive" and case["status"] == "failed"
        for case in summary["cases"]
    )
    any_nonzero_exit = any(
        str(case.get("exit_code") or "0") not in {"0", "None"}
        for case in summary["cases"]
    )
    any_no_victim = any(
        int(case.get("aggregate", {}).get("dispatcher_no_victim_wait_count_total", 0) or 0)
        > 0
        for case in summary["cases"]
    )
    any_all_locked = any(
        int(case.get("aggregate", {}).get("dispatcher_all_locked_event_count_total", 0) or 0)
        > 0
        or int(case.get("log_characterization", {}).get("all_locked_count", 0) or 0) > 0
        for case in summary["cases"]
    )
    any_pending_stall = any(
        int(case.get("aggregate", {}).get("dispatcher_pending_stall_count_total", 0) or 0)
        > 0
        for case in summary["cases"]
    )
    any_prefetch_drop = any(
        int(case.get("aggregate", {}).get("prefetch_drop_count_total", 0) or 0) > 0
        for case in summary["cases"]
    )
    has_dispatcher_pressure = any(
        int(case.get("aggregate", {}).get("dispatcher_eviction_count_total", 0) or 0) > 0
        or int(case.get("aggregate", {}).get("dispatcher_cache_miss_fetch_count_total", 0) or 0)
        > 0
        for case in summary["cases"]
    )
    lines = [
        "# Pressure Canary Summary",
        "",
        f"- Benchmark root: `{summary['benchmark_root']}`",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Trace: `{summary['trace_name']}`",
        "",
        "## Cases",
        "",
        "| Mode | Variant | Status | Exit | f_layers | max_cand | tok/s | mean ms/tok | p95 s | hit | miss | evict | no-victim | pending-stall | all-locked | progress-stall | prefetch drop | cap drop | pressure drop | admit rate | pressure drop rate | under pressure | conflict | locked max | evict min | fatal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        aggregate = case.get("aggregate", {})
        log_info = case.get("log_characterization", {})
        lines.append(
            "| {mode} | {variant} | {status} | {exit_code} | {future_layers} | {max_candidates} | {tps} | {ms_tok} | {p95} | {hit} | {miss} | {evict} | {no_victim} | {pending_stall} | {all_locked} | {progress_stall} | {prefetch_drop} | {cap_drop} | {pressure_drop} | {admit_rate} | {pressure_drop_rate} | {under_pressure} | {conflict} | {locked_max} | {evict_min} | {fatal} |".format(
                mode=case["mode"],
                variant=case["variant"],
                status=case["status"],
                exit_code=case.get("exit_code", "n/a"),
                future_layers=case["prefetch_future_layers"],
                max_candidates=case["prefetch_max_candidates"],
                tps=_fmt(
                    _safe_float(aggregate.get("generated_tokens_per_second")), 3
                ),
                ms_tok=_fmt(
                    _safe_float(
                        aggregate.get("latency_per_generated_token_mean_ms")
                    ),
                    3,
                ),
                p95=_fmt(_safe_float(aggregate.get("latency_p95_s")), 4),
                hit=_fmt(_safe_float(aggregate.get("mean_cache_hit_rate")), 4),
                miss=aggregate.get("dispatcher_cache_miss_fetch_count_total", "n/a"),
                evict=aggregate.get("dispatcher_eviction_count_total", "n/a"),
                no_victim=aggregate.get("dispatcher_no_victim_wait_count_total", "n/a"),
                pending_stall=aggregate.get(
                    "dispatcher_pending_stall_count_total", "n/a"
                ),
                all_locked=aggregate.get(
                    "dispatcher_all_locked_event_count_total",
                    log_info.get("all_locked_count", 0),
                ),
                progress_stall=log_info.get("progress_stall_count", 0),
                prefetch_drop=aggregate.get("prefetch_drop_count_total", "n/a"),
                cap_drop=aggregate.get("prefetch_drop_cap_count_total", "n/a"),
                pressure_drop=aggregate.get(
                    "prefetch_drop_pressure_count_total",
                    "n/a",
                ),
                admit_rate=_fmt(
                    _safe_float(aggregate.get("prefetch_admit_rate")),
                    4,
                ),
                pressure_drop_rate=_fmt(
                    _safe_float(aggregate.get("prefetch_pressure_drop_rate")),
                    4,
                ),
                under_pressure=aggregate.get(
                    "prefetch_under_pressure_count_total",
                    "n/a",
                ),
                conflict=aggregate.get("demand_prefetch_conflict_count_total", "n/a"),
                locked_max=aggregate.get("pressure_locked_max", "n/a"),
                evict_min=aggregate.get("pressure_evictable_min", "n/a"),
                fatal=log_info.get("fatal_count", 0),
            )
        )
    if any_on_demand_failed:
        interpretation = [
            "- On-demand failed, so this pressure point is too strong to isolate prefetch-induced progress issues.",
            "- Lower pressure or add demand progress fixes before using this point as evidence.",
        ]
    elif (
        any_aggressive_failed
        or any_nonzero_exit
        or any_no_victim
        or any_all_locked
        or any_pending_stall
    ):
        interpretation = [
            "- Aggressive prefetch produced a non-complete run or progress events, supporting the hypothesis that speculative expert traffic can violate progress under strong pressure.",
            "- If the exit was external termination, inspect the log and any captured stack artifact before treating it as a runtime fatal.",
            "- Treat this run as a robustness boundary, not a normal performance point.",
        ]
    elif has_dispatcher_pressure and any_prefetch_drop:
        interpretation = [
            "- All cases completed while exercising real expert paging pressure and dropping speculative prefetch traffic.",
            "- This supports the mitigation hypothesis: bounded/best-effort prefetch admission can preserve demand progress under strong pressure.",
            "- Treat this run as post-fix robustness evidence, not as a normal latency-optimization point.",
        ]
    elif has_dispatcher_pressure:
        interpretation = [
            "- All cases completed without no-victim/all-locked events, so this run does not reproduce a progress-boundary failure.",
            "- Dispatcher miss/eviction counters are non-zero, so the run still exercises real expert paging pressure; stronger pressure or longer runs are needed to trigger the boundary.",
        ]
    else:
        interpretation = [
            "- All cases completed and dispatcher pressure counters are zero or unavailable.",
            "- This run is mainly a sanity check; it is not sufficient evidence for paging-pressure behavior.",
        ]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            *interpretation,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root) if args.output_root else benchmark_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = build_summary(benchmark_root)
    (output_root / "pressure_canary_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "pressure_canary_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
