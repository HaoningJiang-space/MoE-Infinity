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
        "fatal_count": text.count("evict_node is nullptr"),
        "failure_marker_count": sum(text.count(marker) for marker in FAILURE_MARKERS),
        "first_all_locked": all_locked_lines[0] if all_locked_lines else None,
        "last_all_locked": all_locked_lines[-1] if all_locked_lines else None,
    }


def _case_status(raw_path: Path, log_info: Mapping[str, Any], exit_code: str | None) -> str:
    if raw_path.is_file() and exit_code in {None, "0"}:
        return "complete"
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
    aggregate = dict(raw.get("aggregate", {}))
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
    cases = [
        _case_summary(
            benchmark_root=benchmark_root,
            mode=mode,
            variant=variant,
            status_rows=status_rows,
        )
        for mode in MODES
        for variant in VARIANTS
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
    lines = [
        "# Pressure Canary Summary",
        "",
        f"- Benchmark root: `{summary['benchmark_root']}`",
        f"- Generated at UTC: `{summary['generated_at_utc']}`",
        f"- Trace: `{summary['trace_name']}`",
        "",
        "## Cases",
        "",
        "| Mode | Variant | Status | Exit | f_layers | max_cand | tok/s | mean ms/tok | p95 s | hit | all-locked | fatal |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        aggregate = case.get("aggregate", {})
        log_info = case.get("log_characterization", {})
        lines.append(
            "| {mode} | {variant} | {status} | {exit_code} | {future_layers} | {max_candidates} | {tps} | {ms_tok} | {p95} | {hit} | {all_locked} | {fatal} |".format(
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
                all_locked=log_info.get("all_locked_count", 0),
                fatal=log_info.get("fatal_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Conservative success plus aggressive failure supports the hypothesis that speculative expert traffic can violate progress under strong pressure.",
            "- If on-demand fails, the pressure point is too strong to isolate prefetch-induced progress issues.",
            "- This run is a robustness canary, not a normal performance sweep.",
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
