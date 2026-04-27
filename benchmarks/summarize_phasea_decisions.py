from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from moe_infinity.analysis.phasea import analyze_phasea_event_file


VARIANT_LABELS = {
    "on_demand": "on_demand",
    "trace_similarity_prefetch": "trace_similarity",
    "history_reuse_backbone": "sequence_object_simple",
    "history_reuse_topk_backbone": "sequence_object_topk",
    "history_reuse_consensus_backbone": "sequence_object_consensus",
    "history_reuse_consensus_backbone_retrieval": "sequence_object_recent_retrieval",
    "history_reuse_local_backbone": "local_continuation_simple",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Qwen benchmark raw metrics and Phase-A event analysis."
    )
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to <benchmark-root>/analysis.",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--budgets", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument(
        "--skip-horizon-analysis",
        action="store_true",
        help="Only compute same-step metrics needed by the decision summary.",
    )
    parser.add_argument(
        "--same-step-budget",
        type=int,
        default=32,
        help="Budget used for compact decision tables.",
    )
    parser.add_argument(
        "--exclude-warmup-events",
        action="store_true",
        help="Ignore policy events emitted for warmup requests.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _pct_delta(new: float, old: float) -> float:
    if old == 0.0:
        return 0.0
    return ((new - old) / old) * 100.0


def _measured_records(raw: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        record
        for record in raw.get("records", [])
        if not bool(record.get("is_warmup", False))
    ]


def _fixed_token_status(raw: Mapping[str, Any]) -> Dict[str, Any]:
    measured = _measured_records(raw)
    generated = [int(record.get("generated_tokens", 0)) for record in measured]
    expected = int(raw.get("max_new_tokens", 0))
    return {
        "expected_generated_tokens": expected,
        "observed_generated_tokens": generated,
        "all_measured_fixed_length": bool(generated)
        and expected > 0
        and all(token_count == expected for token_count in generated),
    }


def _same_step_metrics(
    analysis: Mapping[str, Any] | None,
    *,
    budget: int,
) -> Dict[str, float]:
    if analysis is None:
        return {}
    key = f"M{int(budget)}"
    recall = analysis.get("same_step_window_recall", {}).get(key, {})
    gap = analysis.get("same_step_window_gap_decomposition", {}).get(key, {})
    return {
        "pair_recall": _safe_float(recall.get("mean_pair_recall")),
        "expert_only_recall": _safe_float(recall.get("mean_expert_only_recall")),
        "candidate_omission_gap": _safe_float(
            gap.get("candidate_omission_gap_mean")
        ),
        "restricted_gap": _safe_float(gap.get("restricted_oracle_gap_mean")),
    }


def _compact_case(
    *,
    raw_path: Path,
    benchmark_root: Path,
    horizons: Iterable[int],
    budgets: Iterable[int],
    same_step_budget: int,
    exclude_warmup_events: bool,
) -> Dict[str, Any]:
    raw = _load_json(raw_path)
    trace_name = str(raw.get("trace_name", raw_path.stem.split("__", 1)[0]))
    variant = str(raw.get("variant", raw_path.stem.split("__", 1)[-1]))
    event_path = benchmark_root / "events" / f"{trace_name}__{variant}.jsonl"
    analysis = None
    if event_path.is_file():
        analysis = analyze_phasea_event_file(
            event_path,
            horizons=horizons,
            budgets=budgets,
            exclude_warmup_events=exclude_warmup_events,
        )
    aggregate = dict(raw.get("aggregate", {}))
    phasea = _same_step_metrics(analysis, budget=same_step_budget)
    instrumentation = (
        dict(analysis.get("instrumentation_validation", {})) if analysis else {}
    )
    candidate_health = dict(analysis.get("candidate_health", {})) if analysis else {}
    return {
        "trace_name": trace_name,
        "variant": variant,
        "object_label": VARIANT_LABELS.get(variant, variant),
        "raw_path": str(raw_path),
        "event_path": str(event_path) if event_path.is_file() else None,
        "fixed_tokens": _fixed_token_status(raw),
        "aggregate": aggregate,
        "phasea_same_step": phasea,
        "controller_latency": dict(analysis.get("controller_latency", {}))
        if analysis
        else {},
        "instrumentation_validation": instrumentation,
        "candidate_health": candidate_health,
    }


def _index_cases(cases: list[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    indexed: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for case in cases:
        indexed.setdefault(str(case["trace_name"]), {})[str(case["variant"])] = case
    return indexed


def _comparison(
    trace_name: str,
    lhs: Mapping[str, Any],
    rhs: Mapping[str, Any],
) -> Dict[str, Any]:
    lhs_agg = lhs.get("aggregate", {})
    rhs_agg = rhs.get("aggregate", {})
    lhs_phasea = lhs.get("phasea_same_step", {})
    rhs_phasea = rhs.get("phasea_same_step", {})
    return {
        "trace_name": trace_name,
        "lhs_variant": lhs["variant"],
        "rhs_variant": rhs["variant"],
        "latency_per_token_delta_pct": _pct_delta(
            _safe_float(lhs_agg.get("latency_per_generated_token_mean_ms")),
            _safe_float(rhs_agg.get("latency_per_generated_token_mean_ms")),
        ),
        "tokens_per_second_delta_pct": _pct_delta(
            _safe_float(lhs_agg.get("generated_tokens_per_second")),
            _safe_float(rhs_agg.get("generated_tokens_per_second")),
        ),
        "p95_latency_delta_pct": _pct_delta(
            _safe_float(lhs_agg.get("latency_p95_s")),
            _safe_float(rhs_agg.get("latency_p95_s")),
        ),
        "pair_recall_delta": _safe_float(lhs_phasea.get("pair_recall"))
        - _safe_float(rhs_phasea.get("pair_recall")),
        "expert_recall_delta": _safe_float(lhs_phasea.get("expert_only_recall"))
        - _safe_float(rhs_phasea.get("expert_only_recall")),
        "omission_gap_delta": _safe_float(lhs_phasea.get("candidate_omission_gap"))
        - _safe_float(rhs_phasea.get("candidate_omission_gap")),
        "restricted_gap_delta": _safe_float(lhs_phasea.get("restricted_gap"))
        - _safe_float(rhs_phasea.get("restricted_gap")),
    }


def _build_comparisons(cases: list[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    indexed = _index_cases(cases)
    comparisons: list[Dict[str, Any]] = []
    baselines = [
        "history_reuse_backbone",
        "history_reuse_topk_backbone",
        "history_reuse_consensus_backbone",
        "history_reuse_consensus_backbone_retrieval",
        "on_demand",
    ]
    for trace_name, variants in sorted(indexed.items()):
        local = variants.get("history_reuse_local_backbone")
        if local is None:
            continue
        for baseline in baselines:
            rhs = variants.get(baseline)
            if rhs is None:
                continue
            comparisons.append(_comparison(trace_name, local, rhs))
    return comparisons


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Phase-A Decision Summary",
        "",
        f"- Benchmark root: `{summary['benchmark_root']}`",
        f"- Exclude warmup events: `{str(summary['exclude_warmup_events']).lower()}`",
        f"- Same-step budget: `M{summary['same_step_budget']}`",
        "",
        "## Cases",
        "",
        "| Trace | Variant | Object label | Fixed tokens | tok/s | mean ms/tok | p95 latency (s) | busy waits | evictions | all-locked | no-victim us | event mean us | M32 pair | M32 expert | M32 omission | M32 restricted |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        aggregate = case.get("aggregate", {})
        phasea = case.get("phasea_same_step", {})
        controller = case.get("controller_latency", {})
        fixed = case.get("fixed_tokens", {})
        lines.append(
            "| {trace} | {variant} | {label} | {fixed} | {tps} | {ms_tok} | {p95} | {busy} | {evict} | {all_locked} | {no_victim_us} | {event_us} | {pair} | {expert} | {omit} | {restricted} |".format(
                trace=case["trace_name"],
                variant=case["variant"],
                label=case["object_label"],
                fixed=str(fixed.get("all_measured_fixed_length", False)).lower(),
                tps=_fmt(aggregate.get("generated_tokens_per_second"), 3),
                ms_tok=_fmt(aggregate.get("latency_per_generated_token_mean_ms"), 3),
                p95=_fmt(aggregate.get("latency_p95_s"), 4),
                busy=_fmt(aggregate.get("mean_dispatcher_busy_wait_count"), 2),
                evict=_fmt(aggregate.get("mean_dispatcher_eviction_count"), 2),
                all_locked=_fmt(
                    aggregate.get("mean_dispatcher_all_locked_event_count"), 2
                ),
                no_victim_us=_fmt(
                    aggregate.get("mean_dispatcher_no_victim_wait_total_us"), 1
                ),
                event_us=_fmt(controller.get("decision_latency_mean_us"), 2),
                pair=_fmt(phasea.get("pair_recall"), 4),
                expert=_fmt(phasea.get("expert_only_recall"), 4),
                omit=_fmt(phasea.get("candidate_omission_gap"), 4),
                restricted=_fmt(phasea.get("restricted_gap"), 4),
            )
        )

    lines.extend(
        [
            "",
            "## Local Comparisons",
            "",
            "| Trace | Local vs | ms/tok delta % | tok/s delta % | p95 delta % | pair delta | expert delta | omission delta | restricted delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in summary["comparisons"]:
        lines.append(
            "| {trace} | {rhs} | {ms} | {tps} | {p95} | {pair} | {expert} | {omit} | {restricted} |".format(
                trace=item["trace_name"],
                rhs=item["rhs_variant"],
                ms=_fmt(item["latency_per_token_delta_pct"], 2),
                tps=_fmt(item["tokens_per_second_delta_pct"], 2),
                p95=_fmt(item["p95_latency_delta_pct"], 2),
                pair=_fmt(item["pair_recall_delta"], 4),
                expert=_fmt(item["expert_recall_delta"], 4),
                omit=_fmt(item["omission_gap_delta"], 4),
                restricted=_fmt(item["restricted_gap_delta"], 4),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root) if args.output_root else benchmark_root / "analysis"
    raw_dir = benchmark_root / "raw"
    if not raw_dir.is_dir():
        raise ValueError(f"{benchmark_root} does not contain a raw/ directory.")
    output_root.mkdir(parents=True, exist_ok=True)

    cases = [
        _compact_case(
            raw_path=path,
            benchmark_root=benchmark_root,
            horizons=[] if args.skip_horizon_analysis else args.horizons,
            budgets=args.budgets,
            same_step_budget=args.same_step_budget,
            exclude_warmup_events=args.exclude_warmup_events,
        )
        for path in sorted(raw_dir.glob("*.json"))
    ]
    summary = {
        "benchmark_root": str(benchmark_root),
        "exclude_warmup_events": bool(args.exclude_warmup_events),
        "horizons": [] if args.skip_horizon_analysis else list(args.horizons),
        "budgets": list(args.budgets),
        "skip_horizon_analysis": bool(args.skip_horizon_analysis),
        "same_step_budget": int(args.same_step_budget),
        "cases": cases,
        "comparisons": _build_comparisons(cases),
    }
    (output_root / "decision_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output_root / "decision_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
