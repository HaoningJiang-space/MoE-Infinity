from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from moe_infinity import MoE
from moe_infinity.policies import OffloadingPolicyManager
from moe_infinity.utils.qwen_benchmark import (
    aggregate_request_records,
    build_qwen_benchmark_config,
    count_request_input_tokens,
    diff_counter_dict,
    discover_trace_files,
    load_chat_trace,
)
from benchmarks.benchmark_qwen_offloading import (
    _snapshot_cache_hit_rate,
    _snapshot_dispatcher_stats,
    _snapshot_library_stats,
    _snapshot_prefetcher_stats,
)


ROOT = Path(
    os.environ.get(
        "DEV_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/phasea_dev_inprocess_static_smoke",
    )
)
REPO = Path(os.environ.get("DEV_REPO", "/data/ziheng/projects/moe_infinity_fgo"))
MODEL = Path(
    os.environ.get("DEV_MODEL", "/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat")
)
TRACE_DIR = Path(os.environ.get("DEV_TRACE_DIR", str(REPO / "benchmarks/traces/qwen")))
TRACE_NAME = os.environ.get("DEV_TRACE_NAME", "mixed")
OFFLOAD_CACHE_TEMPLATE = os.environ.get("DEV_OFFLOAD_CACHE_TEMPLATE", "")
RESET_BETWEEN_CASES = os.environ.get("DEV_RESET_BETWEEN_CASES", "1") != "0"
DEV_WARMUP_REQUESTS = int(os.environ.get("DEV_WARMUP_REQUESTS", "0"))
DEV_REPEATS = int(os.environ.get("DEV_REPEATS", "1"))
DEV_BRACKETED_BASELINE = os.environ.get("DEV_BRACKETED_BASELINE", "0") != "0"
DEV_SHUFFLE_CASES = os.environ.get("DEV_SHUFFLE_CASES", "0") != "0"
DEV_RANDOM_SEED = int(os.environ.get("DEV_RANDOM_SEED", "36"))
DEV_DRIFT_INVALID_THRESHOLD = float(
    os.environ.get("DEV_DRIFT_INVALID_THRESHOLD", "0.10")
)


CASES: Dict[str, Dict[str, Any]] = {
    "on_demand": {
        "enable_prefetch": False,
        "policy_disabled": True,
        "execution_mode": "disabled",
    },
    "prefetch_enabled_no_policy": {
        "enable_prefetch": True,
        "policy_disabled": True,
        "execution_mode": "disabled",
    },
    "static_top4_replace_only": {
        "enable_prefetch": True,
        "policy_disabled": False,
        "execution_mode": "replace_only",
    },
    "static_top4_enqueue_only": {
        "enable_prefetch": True,
        "policy_disabled": False,
        "execution_mode": "enqueue_only",
    },
    "static_top4_replace_and_enqueue": {
        "enable_prefetch": True,
        "policy_disabled": False,
        "execution_mode": "replace_and_enqueue",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_case_names() -> List[str]:
    requested = os.environ.get("DEV_CASES")
    if not requested:
        return list(CASES)
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown DEV_CASES entries: {unknown}")
    return names


def _case_sequence_for_repeat(repeat_idx: int) -> List[tuple[str, str]]:
    names = _selected_case_names()
    if not DEV_BRACKETED_BASELINE:
        return [(name, "case") for name in names]

    mechanisms = [name for name in names if name != "on_demand"]
    if DEV_SHUFFLE_CASES:
        random.Random(DEV_RANDOM_SEED + repeat_idx).shuffle(mechanisms)
    return (
        [("on_demand", "baseline_pre")]
        + [(name, "mechanism") for name in mechanisms]
        + [("on_demand", "baseline_post")]
    )


def _apply_case(model: MoE, case: Dict[str, Any]) -> None:
    engine = model.engine
    prefetcher = engine.expert_prefetcher
    prefetcher.prefetch_policy_disabled = bool(case["policy_disabled"])
    prefetcher.prefetch_execution_mode = str(case["execution_mode"])
    prefetcher.prefetch_future_layers = 4
    prefetcher.prefetch_max_candidates = 32
    prefetcher.prefetch_admission_enabled = True
    prefetcher.prefetch_admission_demand_reserve = 2
    prefetcher.prefetch_credit_gated_enabled = True
    prefetcher.prefetch_credit_count = 8
    prefetcher.prefetch_credit_zero_action = "update_only"
    prefetcher.prefetch_retention_protect_demand_eviction = False
    if hasattr(prefetcher.archer_engine, "set_candidate_demand_eviction_protection"):
        prefetcher.archer_engine.set_candidate_demand_eviction_protection(False)
    if hasattr(prefetcher.archer_engine, "replace_cache_candidates"):
        prefetcher.archer_engine.replace_cache_candidates([])
    for module in getattr(engine, "expert_layer_modules", []):
        module.enable_expert_prefetch = bool(case["enable_prefetch"])
        module.expert_policy_score_only = False


def _reset_tracer_runtime_state(model: MoE) -> None:
    tracer = model.engine.expert_tracer
    tracer.trace.clear()
    persistent_capacity = int(getattr(tracer, "persistent_capacity", 0))
    if hasattr(tracer, "trace_collection"):
        tracer.trace_collection[persistent_capacity:] = 0
    if hasattr(tracer, "collection_access"):
        tracer.collection_access[persistent_capacity:] = 0


def _reset_policy_runtime_state(model: MoE) -> None:
    engine = model.engine
    _reset_tracer_runtime_state(model)
    engine.offloading_policy = OffloadingPolicyManager(
        config=engine.archer_config,
        tracer=engine.expert_tracer,
        predictor=engine.expert_predictor,
        model_tag=str(getattr(engine, "model_name", "")).lower(),
    )
    for module in getattr(engine, "expert_layer_modules", []):
        module.expert_policy = engine.offloading_policy


def _reset_runtime_state(model: MoE) -> Dict[str, Any]:
    engine = model.engine
    reset_info: Dict[str, Any] = {"enabled": bool(RESET_BETWEEN_CASES)}
    if not RESET_BETWEEN_CASES:
        reset_info["warning"] = "dirty runtime; GPU residency and policy state are reused"
        return reset_info

    torch.cuda.synchronize()
    if hasattr(engine.archer_engine, "replace_cache_candidates"):
        engine.archer_engine.replace_cache_candidates([])
    if hasattr(engine.expert_dispatcher, "reset_expert_cache_state"):
        engine.expert_dispatcher.reset_expert_cache_state()
        reset_info["expert_cache_reset"] = True
    else:
        reset_info["expert_cache_reset"] = False
        reset_info["warning"] = "dispatcher lacks reset_expert_cache_state"
    if hasattr(engine.expert_dispatcher, "reset_runtime_stats"):
        engine.expert_dispatcher.reset_runtime_stats()
    if hasattr(engine.expert_dispatcher, "clear_expert_cache_counts"):
        engine.expert_dispatcher.clear_expert_cache_counts()
    prefetcher = getattr(engine, "expert_prefetcher", None)
    if prefetcher is not None and hasattr(prefetcher, "reset_prefetch_runtime_stats"):
        prefetcher.reset_prefetch_runtime_stats()
    _reset_policy_runtime_state(model)
    torch.cuda.synchronize()
    return reset_info


def _run_case(
    *,
    model: MoE,
    tokenizer,
    requests,
    run_label: str,
    case_name: str,
    repeat_idx: int,
    case_role: str,
    warmup_requests: int,
    max_input_length: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    case = CASES[case_name]
    reset_info = _reset_runtime_state(model)
    _apply_case(model, case)
    dispatcher = model.engine.expert_dispatcher
    prefetcher = model.engine.expert_prefetcher
    records = []
    previous_library_stats: Dict[str, int] = _snapshot_library_stats(model)
    previous_cache_stats: Dict[str, float | int] = _snapshot_cache_hit_rate(model)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    device = torch.device("cuda:0")

    for index, request in enumerate(requests):
        is_warmup = index < int(warmup_requests)
        if hasattr(dispatcher, "reset_runtime_stats"):
            dispatcher.reset_runtime_stats()
        if hasattr(prefetcher, "reset_prefetch_runtime_stats"):
            prefetcher.reset_prefetch_runtime_stats()
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_token_count = count_request_input_tokens(tokenizer, request.messages)
        encoded = tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        model.engine.phasea_request_context = {
            "request_id": request.request_id,
            "trace_name": TRACE_NAME,
            "tag": request.tag,
            "variant": run_label,
            "is_warmup": is_warmup,
        }

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        latency_s = time.perf_counter() - start
        generated_ids = outputs[0][input_ids.shape[1] :]

        library_stats = _snapshot_library_stats(model)
        library_stats_delta = diff_counter_dict(
            library_stats,
            previous_library_stats,
            keys=(
                "query_count",
                "hit_count",
                "admit_count",
                "duplicate_update_count",
            ),
        )
        cache_hit_rate_snapshot = _snapshot_cache_hit_rate(model)
        cache_hit_rate_delta = diff_counter_dict(
            cache_hit_rate_snapshot,
            previous_cache_stats,
            keys=(
                "visit_count",
                "gpu_visit_count",
                "cpu_visit_count",
                "hit_count",
                "gpu_hit_count",
                "cpu_hit_count",
                "prefetch_count",
            ),
        )
        if cache_hit_rate_delta.get("visit_count", 0) > 0:
            cache_hit_rate_delta["overall_hit_rate"] = (
                cache_hit_rate_delta["hit_count"] / cache_hit_rate_delta["visit_count"]
            )
        else:
            cache_hit_rate_delta["overall_hit_rate"] = 0.0

        records.append(
            {
                "trace_name": TRACE_NAME,
                "variant": run_label,
                "request_index": index,
                "request_id": request.request_id,
                "tag": request.tag,
                "is_warmup": is_warmup,
                "input_tokens": input_token_count,
                "latency_s": latency_s,
                "generated_tokens": int(generated_ids.numel()),
                "latency_per_generated_token_ms": (
                    latency_s * 1000.0 / max(int(generated_ids.numel()), 1)
                ),
                "dispatcher_stats": _snapshot_dispatcher_stats(dispatcher),
                "prefetcher_stats": _snapshot_prefetcher_stats(model),
                "library_stats": library_stats,
                "library_stats_delta": library_stats_delta,
                "cache_hit_rate_snapshot": cache_hit_rate_snapshot,
                "cache_hit_rate_delta": cache_hit_rate_delta,
            }
        )
        previous_library_stats = library_stats
        previous_cache_stats = cache_hit_rate_snapshot

    return {
        "run_label": run_label,
        "case_name": case_name,
        "repeat_idx": int(repeat_idx),
        "case_role": case_role,
        "case": case,
        "warmup_requests": int(warmup_requests),
        "measured_requests": max(len(requests) - int(warmup_requests), 0),
        "reset_info": reset_info,
        "protocol_note": (
            "This harness loads one model process and resets runtime state before each case "
            "when DEV_RESET_BETWEEN_CASES=1. Use it for fast iteration; validate final claims "
            "with a new-process bracket run."
        ),
        "records": records,
        "aggregate": aggregate_request_records(
            [record for record in records if not record["is_warmup"]]
        ),
    }


def _slow_request_count(result: Dict[str, Any], threshold_ms: float = 150.0) -> int:
    return sum(
        1
        for record in result.get("records", [])
        if not record.get("is_warmup", False)
        and float(record.get("latency_per_generated_token_ms", 0.0)) > threshold_ms
    )


def _tps(result: Dict[str, Any]) -> float:
    return float(result.get("aggregate", {}).get("generated_tokens_per_second", 0.0))


def _bracket_info(results: Dict[str, Any]) -> Dict[int, Dict[str, float | bool]]:
    grouped: Dict[int, Dict[str, float]] = {}
    for result in results.values():
        repeat_idx = int(result.get("repeat_idx", 0))
        role = result.get("case_role", "")
        if role not in ("baseline_pre", "baseline_post"):
            continue
        grouped.setdefault(repeat_idx, {})[role] = _tps(result)

    info: Dict[int, Dict[str, float | bool]] = {}
    for repeat_idx, values in grouped.items():
        pre = float(values.get("baseline_pre", 0.0))
        post = float(values.get("baseline_post", 0.0))
        mean = (pre + post) / 2.0 if pre > 0.0 and post > 0.0 else 0.0
        drift = abs(pre - post) / mean if mean > 0.0 else 1.0
        info[repeat_idx] = {
            "baseline_pre_tps": pre,
            "baseline_post_tps": post,
            "baseline_mean_tps": mean,
            "baseline_drift": drift,
            "valid": drift <= DEV_DRIFT_INVALID_THRESHOLD,
        }
    return info


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_normalized_means(
    results: Dict[str, Any],
    brackets: Dict[int, Dict[str, float | bool]],
) -> List[str]:
    grouped: Dict[str, List[float]] = {}
    for result in results.values():
        if result.get("case_role") != "mechanism":
            continue
        repeat_idx = int(result.get("repeat_idx", 0))
        bracket = brackets.get(repeat_idx, {})
        baseline = float(bracket.get("baseline_mean_tps", 0.0))
        if not bracket.get("valid", False) or baseline <= 0.0:
            continue
        grouped.setdefault(str(result.get("case_name", "")), []).append(
            _tps(result) / baseline
        )

    lines = [
        "## Valid Bracket-Normalized Mechanism Means",
        "",
        "| case | valid repeats | mean normalized tok/s |",
        "| --- | ---: | ---: |",
    ]
    for case_name in sorted(grouped):
        values = grouped[case_name]
        lines.append(
            f"| {case_name} | {len(values)} | {_mean(values):.3f} |"
        )
    if not grouped:
        lines.append("| none | 0 | 0.000 |")
    return lines


def _render_summary(results: Dict[str, Any], setup_timing: Dict[str, float]) -> str:
    brackets = _bracket_info(results)
    lines = [
        "# Dev In-Process Static Smoke",
        "",
        "This is a fast iteration protocol. It keeps one loaded model process and, by default, resets expert cache/policy state before each case.",
        "",
        f"- model_load_s: {setup_timing.get('model_load_s', 0.0):.2f}",
        f"- total_setup_s: {setup_timing.get('total_setup_s', 0.0):.2f}",
        f"- reset_between_cases: {str(RESET_BETWEEN_CASES).lower()}",
        f"- warmup_requests_per_case: {DEV_WARMUP_REQUESTS}",
        f"- repeats: {DEV_REPEATS}",
        f"- bracketed_baseline: {str(DEV_BRACKETED_BASELINE).lower()}",
        f"- shuffle_cases: {str(DEV_SHUFFLE_CASES).lower()}",
        f"- random_seed: {DEV_RANDOM_SEED}",
        f"- drift_invalid_threshold: {DEV_DRIFT_INVALID_THRESHOLD:.3f}",
        "",
    ]
    if DEV_BRACKETED_BASELINE:
        lines.extend(_render_normalized_means(results, brackets))
        lines.extend(
            [
                "",
                "## Per-Run Results",
                "",
            ]
        )
    lines.extend(
        [
            "| run | repeat | role | case | tok/s | norm vs bracket | bracket drift | bracket valid | ms/token | slow req >150ms | candidates | admitted | runtime enqueue | queue push | same-device skip | complete | miss | evict |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run_label, result in results.items():
        agg = result.get("aggregate", {})
        repeat_idx = int(result.get("repeat_idx", 0))
        bracket = brackets.get(repeat_idx, {})
        baseline = float(bracket.get("baseline_mean_tps", 0.0))
        tps = float(agg.get("generated_tokens_per_second", 0.0))
        norm = tps / baseline if baseline > 0.0 and result.get("case_role") == "mechanism" else 0.0
        drift = float(bracket.get("baseline_drift", 0.0))
        valid = bracket.get("valid", True)
        lines.append(
            "| {run} | {repeat} | {role} | {case} | {tps:.3f} | {norm:.3f} | {drift:.3f} | {valid} | {mpt:.2f} | {slow} | {cand} | {admit} | {rt_enq} | {push} | {skip} | {comp} | {miss} | {evict} |".format(
                run=run_label,
                repeat=repeat_idx,
                role=result.get("case_role", ""),
                case=result.get("case_name", ""),
                tps=tps,
                norm=norm,
                drift=drift,
                valid=str(bool(valid)).lower(),
                mpt=float(agg.get("latency_per_generated_token_mean_ms", 0.0)),
                slow=_slow_request_count(result),
                cand=agg.get("prefetch_candidate_count_total", 0),
                admit=agg.get("prefetch_admitted_count_total", 0),
                rt_enq=agg.get("prefetch_runtime_enqueue_count_total", 0),
                push=agg.get("prefetch_runtime_queue_push_count_total", 0),
                skip=agg.get("prefetch_runtime_same_device_skip_count_total", 0),
                comp=agg.get("prefetch_runtime_complete_count_total", 0),
                miss=agg.get("dispatcher_cache_miss_fetch_count_total", 0),
                evict=agg.get("dispatcher_eviction_count_total", 0),
            )
        )
    return "\n".join(lines)


def main() -> None:
    if not OFFLOAD_CACHE_TEMPLATE:
        raise ValueError("Set DEV_OFFLOAD_CACHE_TEMPLATE to a prepared offload cache.")
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "raw").mkdir(exist_ok=True)
    (ROOT / "analysis").mkdir(exist_ok=True)
    (ROOT / "driver.log").write_text("", encoding="utf-8")

    trace_files = discover_trace_files(TRACE_DIR)
    trace_path = trace_files[TRACE_NAME]
    measured_requests = int(os.environ.get("DEV_MEASURED_REQUESTS", "2"))
    max_input_length = int(os.environ.get("DEV_MAX_INPUT_LENGTH", "128"))
    max_new_tokens = int(os.environ.get("DEV_MAX_NEW_TOKENS", "8"))
    request_count = DEV_WARMUP_REQUESTS + measured_requests
    requests = load_chat_trace(trace_path)[:request_count]

    setup_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=True,
        use_fast=False,
    )
    config = build_qwen_benchmark_config(
        variant="static_hot_prefetch",
        offload_path=OFFLOAD_CACHE_TEMPLATE,
        device_memory_ratio=float(os.environ.get("DEV_DEVICE_MEMORY_RATIO", "0.30")),
        num_threads=1,
        library_capacity=32,
        library_metric="cosine",
        library_admission="diversity_aware",
        prefetch_future_layers=4,
        prefetch_max_candidates=32,
        prefetch_admission_enabled=True,
        prefetch_credit_gated_enabled=True,
        prefetch_credit_count=8,
        prefetch_execution_mode="replace_only",
        static_prefetch_default_topk=4,
    )
    model_start = time.perf_counter()
    model = MoE(str(MODEL), config)
    setup_timing = {
        "model_load_s": time.perf_counter() - model_start,
        "total_setup_s": time.perf_counter() - setup_start,
    }

    results: Dict[str, Any] = {}
    try:
        for repeat_idx in range(max(DEV_REPEATS, 1)):
            for case_idx, (case_name, case_role) in enumerate(
                _case_sequence_for_repeat(repeat_idx)
            ):
                run_label = (
                    f"r{repeat_idx:02d}__c{case_idx:02d}__{case_role}__{case_name}"
                )
                with (ROOT / "driver.log").open("a", encoding="utf-8") as log:
                    log.write(f"[{_now()}] start {run_label}\n")
                results[run_label] = _run_case(
                    model=model,
                    tokenizer=tokenizer,
                    requests=requests,
                    run_label=run_label,
                    case_name=case_name,
                    repeat_idx=repeat_idx,
                    case_role=case_role,
                    warmup_requests=DEV_WARMUP_REQUESTS,
                    max_input_length=max_input_length,
                    max_new_tokens=max_new_tokens,
                )
                with (ROOT / "driver.log").open("a", encoding="utf-8") as log:
                    log.write(f"[{_now()}] done {run_label}\n")
    finally:
        del model
        torch.cuda.empty_cache()

    payload = {
        "setup_timing": setup_timing,
        "offload_cache_template": OFFLOAD_CACHE_TEMPLATE,
        "reset_between_cases": RESET_BETWEEN_CASES,
        "warmup_requests": DEV_WARMUP_REQUESTS,
        "measured_requests": measured_requests,
        "repeats": DEV_REPEATS,
        "bracketed_baseline": DEV_BRACKETED_BASELINE,
        "shuffle_cases": DEV_SHUFFLE_CASES,
        "random_seed": DEV_RANDOM_SEED,
        "drift_invalid_threshold": DEV_DRIFT_INVALID_THRESHOLD,
        "brackets": _bracket_info(results),
        "results": results,
    }
    (ROOT / "analysis" / "dev_inprocess_static_smoke.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (ROOT / "analysis" / "dev_inprocess_static_smoke.md").write_text(
        _render_summary(results, setup_timing),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
