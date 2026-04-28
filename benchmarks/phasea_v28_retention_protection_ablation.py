from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(
    os.environ.get(
        "V28_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/phasea_v28_retention_protection_ablation",
    )
)
REPO = Path(os.environ.get("V28_REPO", "/data/ziheng/projects/moe_infinity_fgo"))
PYTHON = Path(
    os.environ.get("V28_PYTHON", "/home/ziheng/miniconda3/envs/mxmoe/bin/python")
)
MODEL = Path(
    os.environ.get("V28_MODEL", "/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat")
)
TRACE_DIR = Path(os.environ.get("V28_TRACE_DIR", str(REPO / "benchmarks/traces/qwen")))
CUDA_VISIBLE_DEVICES = os.environ.get("V28_CUDA_VISIBLE_DEVICES", "1")
TRACE_NAME = os.environ.get("V28_TRACE_NAME", "mixed")
TIMEOUT_S = int(os.environ.get("V28_TIMEOUT_S", "1200"))
OFFLOAD_CACHE_TEMPLATE = os.environ.get("V28_OFFLOAD_CACHE_TEMPLATE", "")
OFFLOAD_CACHE_MODE = os.environ.get("V28_OFFLOAD_CACHE_MODE", "fresh")


CASES: Dict[str, Dict[str, Any]] = {
    "on_demand": {
        "variant": "on_demand",
        "future_layers": 0,
        "max_candidates": 0,
        "static_topk": 8,
        "admission": False,
        "credit_gated": False,
        "credit_count": -1,
        "execution_mode": "replace_and_enqueue",
        "protect": False,
    },
    "static_top4_replace_only_no_protect": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 4,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "execution_mode": "replace_only",
        "protect": False,
    },
    "static_top4_replace_only_with_protect": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 4,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "execution_mode": "replace_only",
        "protect": True,
    },
    "static_top4_enqueue_only": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 4,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "execution_mode": "enqueue_only",
        "protect": False,
    },
    "static_top4_replace_and_enqueue_no_protect": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 4,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "execution_mode": "replace_and_enqueue",
        "protect": False,
    },
    "static_top4_replace_and_enqueue_with_protect": {
        "variant": "static_hot_prefetch",
        "future_layers": 4,
        "max_candidates": 32,
        "static_topk": 4,
        "admission": True,
        "credit_gated": True,
        "credit_count": 8,
        "execution_mode": "replace_and_enqueue",
        "protect": True,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_case_names() -> List[str]:
    requested = os.environ.get("V28_CASES")
    if not requested:
        return list(CASES)
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown V28_CASES entries: {unknown}")
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
        os.environ.get("V28_WARMUP_REQUESTS", "2"),
        "--measured-requests",
        os.environ.get("V28_MEASURED_REQUESTS", "16"),
        "--max-new-tokens",
        os.environ.get("V28_MAX_NEW_TOKENS", "16"),
        "--fixed-new-tokens",
        "--max-input-length",
        os.environ.get("V28_MAX_INPUT_LENGTH", "128"),
        "--device-memory-ratio",
        os.environ.get("V28_DEVICE_MEMORY_RATIO", "0.30"),
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
        "--prefetch-execution-mode",
        str(case["execution_mode"]),
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
    if OFFLOAD_CACHE_TEMPLATE:
        cmd.extend(
            [
                "--offload-cache-template",
                OFFLOAD_CACHE_TEMPLATE,
                "--offload-cache-mode",
                OFFLOAD_CACHE_MODE,
            ]
        )
    if bool(case["admission"]):
        cmd.append("--prefetch-admission-enabled")
    if bool(case["credit_gated"]):
        cmd.append("--prefetch-credit-gated-enabled")
    if bool(case["protect"]):
        cmd.append("--prefetch-retention-protect-demand-eviction")
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
    loaded["execution_mode"] = case["execution_mode"]
    loaded["retention_protect"] = bool(case["protect"])
    shutil.rmtree(case_root / "offload", ignore_errors=True)
    return loaded


def _render_summary(results: Dict[str, Any]) -> str:
    lines = [
        "# V28 Retention Protection Ablation",
        "",
        "Purpose: test whether explicit demand-path candidate protection makes retention a real mechanism, separate from H2D prefetch.",
        "",
        "| case | mode | protect | rc | failed | tok/s | ms/token | candidates | admitted | runtime enqueue | queue push | same-device skip | dequeue | complete | candidate hit | candidate miss | protect skip | protect fallback | resident hit | late miss | miss | evict | stall |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        agg = result.get("aggregate", {})
        lines.append(
            "| {name} | {mode} | {protect} | {rc} | {failed} | {tps:.3f} | {mpt:.2f} | {cand} | {admit} | {rt_enq} | {rt_push} | {same_dev} | {rt_deq} | {rt_comp} | {cand_hit} | {cand_miss} | {protect_skip} | {protect_fallback} | {resident} | {late} | {miss} | {evict} | {stall} |".format(
                name=name,
                mode=result.get("execution_mode", ""),
                protect=str(bool(result.get("retention_protect", False))).lower(),
                rc=result.get("returncode", ""),
                failed=str(bool(result.get("failed", False))).lower(),
                tps=float(agg.get("generated_tokens_per_second", 0.0)),
                mpt=float(agg.get("latency_per_generated_token_mean_ms", 0.0)),
                cand=agg.get("prefetch_candidate_count_total", 0),
                admit=agg.get("prefetch_admitted_count_total", 0),
                rt_enq=agg.get("prefetch_runtime_enqueue_count_total", 0),
                rt_push=agg.get("prefetch_runtime_queue_push_count_total", 0),
                same_dev=agg.get(
                    "prefetch_runtime_same_device_skip_count_total", 0
                ),
                rt_deq=agg.get("prefetch_runtime_dequeue_count_total", 0),
                rt_comp=agg.get("prefetch_runtime_complete_count_total", 0),
                cand_hit=agg.get("dispatcher_candidate_resident_hit_count_total", 0),
                cand_miss=agg.get("dispatcher_candidate_demand_miss_count_total", 0),
                protect_skip=agg.get(
                    "dispatcher_demand_candidate_protect_skip_count_total", 0
                ),
                protect_fallback=agg.get(
                    "dispatcher_demand_candidate_protect_fallback_count_total", 0
                ),
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
            "Decision rules:",
            "",
            "- `replace_only_with_protect` reducing miss/evict would validate retention/protection as a first-class mechanism.",
            "- `enqueue_only` with H2D completes but no candidate/resident hits means true H2D prefetch is not timely/useful in this setup.",
            "- Protection fallback must stay bounded; otherwise retention can harm demand progress.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Run the V28 retention-protection ablation.")
        print("Configuration is controlled by V28_* environment variables.")
        print("Cases:")
        for name in CASES:
            print(f"  {name}")
        return

    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "driver.log").write_text("", encoding="utf-8")

    results: Dict[str, Any] = {}
    for name in _selected_case_names():
        with (ROOT / "driver.log").open("a", encoding="utf-8") as log:
            log.write(f"[{_now()}] start {name}\n")
        result = _run_case(name)
        results[name] = result
        with (ROOT / "driver.log").open("a", encoding="utf-8") as log:
            log.write(
                f"[{_now()}] done {name} rc={result.get('returncode')}\n"
            )

    summary_json = ROOT / "analysis" / "v28_retention_protection_ablation.json"
    summary_md = ROOT / "analysis" / "v28_retention_protection_ablation.md"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_md.write_text(_render_summary(results), encoding="utf-8")


if __name__ == "__main__":
    main()
