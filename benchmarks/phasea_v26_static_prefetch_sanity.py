from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(
    os.environ.get(
        "V26_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/phasea_v26_static_prefetch_sanity",
    )
)
REPO = Path(os.environ.get("V26_REPO", "/data/ziheng/projects/moe_infinity_fgo"))
PYTHON = Path(
    os.environ.get("V26_PYTHON", "/home/ziheng/miniconda3/envs/mxmoe/bin/python")
)
MODEL = Path(
    os.environ.get("V26_MODEL", "/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat")
)
TRACE_DIR = Path(os.environ.get("V26_TRACE_DIR", str(REPO / "benchmarks/traces/qwen")))
CUDA_VISIBLE_DEVICES = os.environ.get("V26_CUDA_VISIBLE_DEVICES", "1")
TRACE_NAME = os.environ.get("V26_TRACE_NAME", "mixed")
TIMEOUT_S = int(os.environ.get("V26_TIMEOUT_S", "1200"))


CASES: Dict[str, Dict[str, Any]] = {
    "on_demand": {
        "variant": "on_demand",
        "future_layers": 0,
        "max_candidates": 0,
        "static_topk": 8,
        "admission": False,
        "credit_gated": False,
        "credit_count": -1,
        "policy_disabled": False,
    },
    "prefetch_enabled_no_policy": {
        "variant": "history_reuse_local_backbone",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 8,
        "admission": True,
        "credit_gated": False,
        "credit_count": -1,
        "policy_disabled": True,
    },
    "static_hot_top4": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 4,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "policy_disabled": False,
    },
    "static_hot_top8": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 8,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "policy_disabled": False,
    },
    "local_sync_cap8": {
        "variant": "history_reuse_local_backbone",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 8,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "policy_disabled": False,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_case_names() -> List[str]:
    requested = os.environ.get("V26_CASES")
    if not requested:
        return list(CASES)
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown V26_CASES entries: {unknown}")
    return names


def _cmd(case_root: Path, case: Dict[str, Any]) -> List[str]:
    cmd = [
        str(PYTHON),
        "benchmarks/benchmark_qwen_offloading.py",
        "--model-path",
        str(MODEL),
        "--output-root",
        str(case_root),
        "--trace-dir",
        str(TRACE_DIR),
        "--variants",
        str(case["variant"]),
        "--traces",
        TRACE_NAME,
        "--warmup-requests",
        os.environ.get("V26_WARMUP_REQUESTS", "2"),
        "--measured-requests",
        os.environ.get("V26_MEASURED_REQUESTS", "16"),
        "--max-new-tokens",
        os.environ.get("V26_MAX_NEW_TOKENS", "16"),
        "--fixed-new-tokens",
        "--max-input-length",
        os.environ.get("V26_MAX_INPUT_LENGTH", "128"),
        "--device-memory-ratio",
        os.environ.get("V26_DEVICE_MEMORY_RATIO", "0.30"),
        "--num-threads",
        "1",
        "--library-capacity",
        "32",
        "--library-metric",
        "cosine",
        "--backbone-topk",
        "8",
        "--prefetch-future-layers",
        str(case["future_layers"]),
        "--prefetch-max-candidates",
        str(case["max_candidates"]),
        "--prefetch-admission-demand-reserve",
        "2",
        "--prefetch-admission-locked-ratio-threshold",
        "0.8",
        "--prefetch-admission-max-under-pressure",
        "4",
        "--prefetch-admission-max-per-plan",
        "-1",
        "--prefetch-credit-count",
        str(case["credit_count"]),
        "--prefetch-credit-zero-action",
        "update_only",
        "--static-prefetch-default-topk",
        str(case["static_topk"]),
        "--historical-reuse-match-topk",
        "4",
        "--historical-reuse-match-min-required",
        "2",
        "--local-continuation-library-capacity",
        "4096",
        "--local-continuation-key-layers",
        "4",
        "--local-continuation-future-layers",
        "4",
        "--local-continuation-match-topk",
        "4",
        "--local-continuation-match-min-required",
        "2",
    ]
    if bool(case["admission"]):
        cmd.append("--prefetch-admission-enabled")
    if bool(case["credit_gated"]):
        cmd.append("--prefetch-credit-gated-enabled")
    if bool(case["policy_disabled"]):
        cmd.append("--prefetch-policy-disabled")
    return cmd


def _case_raw(case_root: Path, case: Dict[str, Any]) -> Path:
    return case_root / "raw" / f"{TRACE_NAME}__{case['variant']}.json"


def _load_result(case_root: Path, case: Dict[str, Any]) -> Dict[str, Any]:
    path = _case_raw(case_root, case)
    if not path.exists():
        return {"missing_raw": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _run_case(name: str) -> Dict[str, Any]:
    case = CASES[name]
    case_root = ROOT / name
    case_root.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/home/ziheng/miniconda3/envs/mxmoe/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(REPO),
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "TMPDIR": "/data/ziheng/tmp",
            "TEMP": "/data/ziheng/tmp",
            "TMP": "/data/ziheng/tmp",
        }
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            result = subprocess.run(
                _cmd(case_root, case),
                cwd=REPO,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
            )
            returncode = int(result.returncode)
        except subprocess.TimeoutExpired:
            returncode = 124
            log_file.write(f"\nTIMEOUT after {TIMEOUT_S}s\n")
    loaded = _load_result(case_root, case)
    loaded["case_name"] = name
    loaded["returncode"] = returncode
    shutil.rmtree(case_root / "offload", ignore_errors=True)
    return loaded


def _render_summary(results: Dict[str, Any]) -> str:
    lines = [
        "# V26 Static Prefetch Sanity",
        "",
        "Purpose: separate no-sync prefetch data-plane behavior from synchronous local-continuation trace capture.",
        "",
        "| case | rc | failed | tok/s | ms/token | runtime enqueue | queue push | same-device skip | runtime dequeue | runtime complete | queue cleared | resident hit | late miss | miss | evict | stall |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        agg = result.get("aggregate", {})
        lines.append(
            "| {name} | {rc} | {failed} | {tps:.3f} | {mpt:.2f} | {rt_enq} | {rt_push} | {same_dev} | {rt_deq} | {rt_comp} | {cleared} | {resident} | {late} | {miss} | {evict} | {stall} |".format(
                name=name,
                rc=result.get("returncode", ""),
                failed=str(bool(result.get("failed", False))).lower(),
                tps=float(agg.get("generated_tokens_per_second", 0.0)),
                mpt=float(agg.get("latency_per_generated_token_mean_ms", 0.0)),
                rt_enq=agg.get("prefetch_runtime_enqueue_count_total", 0),
                rt_push=agg.get("prefetch_runtime_queue_push_count_total", 0),
                same_dev=agg.get(
                    "prefetch_runtime_same_device_skip_count_total", 0
                ),
                rt_deq=agg.get("prefetch_runtime_dequeue_count_total", 0),
                rt_comp=agg.get("prefetch_runtime_complete_count_total", 0),
                cleared=agg.get("prefetch_runtime_queue_cleared_task_count_total", 0),
                resident=agg.get("dispatcher_prefetch_resident_hit_count_total", 0),
                late=agg.get("dispatcher_late_prefetch_demand_miss_count_total", 0),
                miss=agg.get("dispatcher_cache_miss_fetch_count_total", 0),
                evict=agg.get("dispatcher_eviction_count_total", 0),
                stall=agg.get("dispatcher_pending_stall_count_total", 0),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- If static cases complete and show runtime complete/resident-hit counters, the prefetch data path is alive without sync trace capture.",
            "- If static cases enqueue but queue-push/dequeue/complete stay near zero, they are retention/candidate-set effects, not real H2D prefetch hits.",
            "- If static cases stay near on-demand but local_sync_cap8 remains slow, the bottleneck is the synchronous policy/control path.",
            "- If static cases enqueue but have high queue-cleared or late-miss counts, the lifecycle/cancellation semantics are the next target.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "driver.log").write_text("", encoding="utf-8")
    selected = _selected_case_names()
    results: Dict[str, Any] = {}
    for name in selected:
        with (ROOT / "driver.log").open("a", encoding="utf-8") as handle:
            handle.write(f"[{_now()}] start {name}\n")
        results[name] = _run_case(name)
        with (ROOT / "driver.log").open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{_now()}] done {name} rc={results[name].get('returncode')}\n"
            )
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "v26_static_prefetch_sanity.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (analysis / "v26_static_prefetch_sanity.md").write_text(
        _render_summary(results),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
