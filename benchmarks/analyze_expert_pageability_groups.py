from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


Pair = Tuple[int, int]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _load_pairs(path: Path, metric: str) -> Dict[int, Dict[Pair, Dict[str, float]]]:
    layers: Dict[int, Dict[Pair, Dict[str, float]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if metric not in (reader.fieldnames or []):
            raise ValueError(f"Metric {metric!r} not found in {path}; fields={reader.fieldnames}")
        for row in reader:
            layer = int(row["layer"])
            i = int(row["expert_i"])
            j = int(row["expert_j"])
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            layers[layer][key] = {
                name: float(value)
                for name, value in row.items()
                if name not in {"layer", "expert_i", "expert_j"} and value not in ("", None)
            }
    return dict(layers)


def _experts(pair_metrics: Dict[Pair, Dict[str, float]]) -> List[int]:
    result = set()
    for i, j in pair_metrics:
        result.add(i)
        result.add(j)
    return sorted(result)


def _metric(pair_metrics: Dict[Pair, Dict[str, float]], i: int, j: int, metric: str) -> float:
    if i == j:
        return 1.0
    key = (min(i, j), max(i, j))
    return float(pair_metrics[key][metric])


def _centrality(
    pair_metrics: Dict[Pair, Dict[str, float]],
    candidates: List[int],
    metric: str,
    lower_is_better: bool,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for expert in candidates:
        vals = [
            _metric(pair_metrics, expert, other, metric)
            for other in candidates
            if other != expert
        ]
        score = _mean(vals)
        scores[expert] = -score if lower_is_better else score
    return scores


def _greedy_groups(
    pair_metrics: Dict[Pair, Dict[str, float]],
    group_size: int,
    metric: str,
    lower_is_better: bool,
) -> List[List[int]]:
    unassigned = set(_experts(pair_metrics))
    groups: List[List[int]] = []
    while unassigned:
        candidates = sorted(unassigned)
        centrality = _centrality(pair_metrics, candidates, metric, lower_is_better)
        seed = max(candidates, key=lambda item: (centrality[item], -item))
        group = [seed]
        unassigned.remove(seed)
        while unassigned and len(group) < group_size:
            ranked = []
            for expert in sorted(unassigned):
                vals = [_metric(pair_metrics, expert, member, metric) for member in group]
                score = _mean(vals)
                ranked.append(((-score if lower_is_better else score), -expert, expert))
            chosen = max(ranked)[2]
            group.append(chosen)
            unassigned.remove(chosen)
        groups.append(sorted(group))
    return groups


def _group_stats(
    pair_metrics: Dict[Pair, Dict[str, float]],
    groups: List[List[int]],
    metric: str,
    lower_is_better: bool,
) -> List[Dict[str, Any]]:
    global_values = [float(metrics[metric]) for metrics in pair_metrics.values()]
    global_mean = _mean(global_values)
    result: List[Dict[str, Any]] = []
    for group_id, group in enumerate(groups):
        values = []
        for idx, i in enumerate(group):
            for j in group[idx + 1 :]:
                values.append(_metric(pair_metrics, i, j, metric))
        internal_mean = _mean(values)
        lift = (global_mean - internal_mean) if lower_is_better else (internal_mean - global_mean)
        result.append(
            {
                "group_id": group_id,
                "experts": group,
                "pair_count": len(values),
                "metric_mean": internal_mean,
                "metric_min": min(values) if values else 0.0,
                "metric_max": max(values) if values else 0.0,
                "global_mean": global_mean,
                "lift_vs_global": lift,
            }
        )
    return result


def _summarize_layer(
    layer: int,
    pair_metrics: Dict[Pair, Dict[str, float]],
    group_sizes: List[int],
    metric: str,
    lower_is_better: bool,
) -> Dict[str, Any]:
    pair_values = [float(metrics[metric]) for metrics in pair_metrics.values()]
    layer_result: Dict[str, Any] = {
        "layer": layer,
        "experts": _experts(pair_metrics),
        "pair_count": len(pair_values),
        "global_metric_mean": _mean(pair_values),
        "global_metric_median": statistics.median(pair_values) if pair_values else 0.0,
        "groups_by_size": {},
    }
    for group_size in group_sizes:
        groups = _greedy_groups(pair_metrics, group_size, metric, lower_is_better)
        stats = _group_stats(pair_metrics, groups, metric, lower_is_better)
        layer_result["groups_by_size"][str(group_size)] = {
            "group_count": len(groups),
            "mean_lift_vs_global": _mean(item["lift_vs_global"] for item in stats),
            "mean_internal_metric": _mean(item["metric_mean"] for item in stats),
            "groups": stats,
        }
    return layer_result


def _write_csv(path: Path, layers: List[Dict[str, Any]], group_sizes: List[int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer",
                "group_size",
                "group_id",
                "experts",
                "pair_count",
                "metric_mean",
                "metric_min",
                "metric_max",
                "global_mean",
                "lift_vs_global",
            ],
        )
        writer.writeheader()
        for layer in layers:
            for group_size in group_sizes:
                for group in layer["groups_by_size"][str(group_size)]["groups"]:
                    writer.writerow(
                        {
                            "layer": layer["layer"],
                            "group_size": group_size,
                            "group_id": group["group_id"],
                            "experts": " ".join(str(item) for item in group["experts"]),
                            "pair_count": group["pair_count"],
                            "metric_mean": group["metric_mean"],
                            "metric_min": group["metric_min"],
                            "metric_max": group["metric_max"],
                            "global_mean": group["global_mean"],
                            "lift_vs_global": group["lift_vs_global"],
                        }
                    )


def _write_markdown(
    path: Path,
    *,
    input_csv: Path,
    metric: str,
    lower_is_better: bool,
    layers: List[Dict[str, Any]],
    group_sizes: List[int],
) -> None:
    lines = [
        "# Expert Pageability Groups",
        "",
        f"- Input: `{input_csv}`",
        f"- Metric: `{metric}` ({'lower is better' if lower_is_better else 'higher is better'})",
        "",
        "This is an offline grouping analysis. It does not change routing or",
        "replace experts; it only proposes functionally coherent paging groups.",
        "",
        "| layer | pairs | global mean | group size | mean internal | mean lift | group count |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for layer in layers:
        for group_size in group_sizes:
            summary = layer["groups_by_size"][str(group_size)]
            lines.append(
                "| {layer} | {pairs} | {global_mean:.6f} | {group_size} | {internal:.6f} | {lift:.6f} | {count} |".format(
                    layer=layer["layer"],
                    pairs=layer["pair_count"],
                    global_mean=layer["global_metric_mean"],
                    group_size=group_size,
                    internal=summary["mean_internal_metric"],
                    lift=summary["mean_lift_vs_global"],
                    count=summary["group_count"],
                )
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Positive lift means grouping found experts more similar than the layer-wide average.",
            "- A useful paging mechanism should next show fewer misses/evictions or lower transferred bytes under the same memory ratio.",
            "- If group lift is weak, functionally clustered paging is unlikely to be a strong mechanism for this model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build functionally coherent expert paging groups.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path(
            "moe_infinity_fgo_runs/subspace_similarity_v1/"
            "deepseek_v2_lite_functional_gsm8k1024_len256_tok4096/"
            "expert_pair_functional_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("moe_infinity_fgo_runs/expert_pageability_groups_v1"),
    )
    parser.add_argument("--metric", default="frobenius_similarity")
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--group-sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--layers", nargs="+", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    by_layer = _load_pairs(args.input_csv, args.metric)
    selected_layers = sorted(by_layer)
    if args.layers:
        requested = set(args.layers)
        selected_layers = [layer for layer in selected_layers if layer in requested]
    layers = [
        _summarize_layer(
            layer,
            by_layer[layer],
            args.group_sizes,
            args.metric,
            bool(args.lower_is_better),
        )
        for layer in selected_layers
    ]
    payload = {
        "timestamp_utc": _now(),
        "input_csv": str(args.input_csv),
        "metric": args.metric,
        "lower_is_better": bool(args.lower_is_better),
        "group_sizes": args.group_sizes,
        "layers": layers,
    }
    (args.output_root / "expert_pageability_groups.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(args.output_root / "expert_pageability_groups.csv", layers, args.group_sizes)
    _write_markdown(
        args.output_root / "expert_pageability_groups.md",
        input_csv=args.input_csv,
        metric=args.metric,
        lower_is_better=bool(args.lower_is_better),
        layers=layers,
        group_sizes=args.group_sizes,
    )
    print(args.output_root / "expert_pageability_groups.md")


if __name__ == "__main__":
    main()
