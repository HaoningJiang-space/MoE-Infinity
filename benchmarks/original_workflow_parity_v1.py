from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/original_workflow_parity_v1",
    )
)
FGO_REPO = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_FGO_REPO",
        "/data/ziheng/projects/moe_infinity_fgo",
    )
)
UPSTREAM_REPO = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_UPSTREAM_REPO",
        "/data/ziheng/projects/MoE-Infinity",
    )
)
PYTHON = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_PYTHON",
        "/home/ziheng/miniconda3/envs/mxmoe/bin/python",
    )
)
FGO_MODEL = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_FGO_MODEL",
        "/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat",
    )
)
UPSTREAM_MODEL = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_UPSTREAM_MODEL",
        "/data/ziheng/models/DeepSeek-V2-Lite",
    )
)
TRACE_DIR = Path(
    os.environ.get(
        "ORIGINAL_WORKFLOW_TRACE_DIR",
        str(FGO_REPO / "benchmarks/traces/qwen"),
    )
)
CUDA_VISIBLE_DEVICES = os.environ.get("ORIGINAL_WORKFLOW_CUDA_VISIBLE_DEVICES", "0")
TIMEOUT_S = int(os.environ.get("ORIGINAL_WORKFLOW_TIMEOUT_S", "1800"))


CASES: Dict[str, Dict[str, Any]] = {
    "upstream_readme_default": {
        "kind": "upstream",
        "prefetch_flag": None,
    },
    "upstream_prefetch_flag": {
        "kind": "upstream",
        "prefetch_flag": True,
    },
    "fgo_on_demand": {
        "kind": "fgo",
        "variant": "on_demand",
        "prefetch": False,
        "execution_mode": "disabled",
        "protect": False,
    },
    "fgo_static_replace_enqueue_with_protect": {
        "kind": "fgo",
        "variant": "static_hot_prefetch",
        "prefetch": True,
        "execution_mode": "replace_and_enqueue",
        "protect": True,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_cases(args: argparse.Namespace) -> List[str]:
    if args.cases:
        names = args.cases
    else:
        names = [
            item.strip()
            for item in os.environ.get("ORIGINAL_WORKFLOW_CASES", "").split(",")
            if item.strip()
        ]
    if not names:
        names = list(CASES)
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    return names


def _env(repo: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "PYTHONPATH": str(repo),
            "PATH": "/home/ziheng/miniconda3/envs/mxmoe/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin",
            "TMPDIR": "/data/ziheng/tmp",
            "TEMP": "/data/ziheng/tmp",
            "TMP": "/data/ziheng/tmp",
        }
    )
    return env


def _write_upstream_runner(path: Path) -> None:
    path.write_text(_UPSTREAM_RUNNER, encoding="utf-8")


_UPSTREAM_RUNNER = r'''
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

import moe_infinity.memory.expert_prefetcher as expert_prefetcher_mod


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--offload-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-memory-ratio", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--prefetch-flag", choices=("unset", "true", "false"), default="unset")
    parser.add_argument("--prompt", default="Explain expert offloading for MoE inference.")
    args = parser.parse_args()

    counters = {
        "prefetch_experts_calls": 0,
        "prefetch_experts_list_calls": 0,
        "fetch_experts_lock_cache_calls": 0,
    }

    def wrap(method_name: str, counter_name: str) -> None:
        original = getattr(expert_prefetcher_mod.ExpertPrefetcher, method_name)

        def wrapped(self, *a, **kw):
            counters[counter_name] += 1
            return original(self, *a, **kw)

        setattr(expert_prefetcher_mod.ExpertPrefetcher, method_name, wrapped)

    wrap("prefetch_experts", "prefetch_experts_calls")
    wrap("prefetch_experts_list", "prefetch_experts_list_calls")
    wrap("fetch_experts_lock_cache", "fetch_experts_lock_cache_calls")

    result = {
        "model": args.model,
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
        "max_new_tokens": args.max_new_tokens,
        "prefetch_flag": args.prefetch_flag,
        "success": False,
        "prefetch_counters": counters,
    }
    try:
        from moe_infinity import MoE

        config = {
            "offload_path": args.offload_path,
            "device_memory_ratio": args.device_memory_ratio,
        }
        if args.prefetch_flag != "unset":
            config["prefetch"] = args.prefetch_flag == "true"

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token

        setup_start = time.time()
        model = MoE(args.model, config)
        setup_s = time.time() - setup_start

        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to("cuda:0")
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        elapsed_s = time.time() - start
        generated = max(int(output_ids.shape[-1] - input_ids.shape[-1]), 0)
        result.update(
            {
                "success": True,
                "setup_s": setup_s,
                "generate_s": elapsed_s,
                "generated_tokens": generated,
                "generated_tokens_per_second": generated / elapsed_s if elapsed_s > 0 else 0.0,
                "prefetch_counters": counters,
            }
        )
    except Exception as exc:
        result.update(
            {
                "success": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=20),
                "prefetch_counters": counters,
            }
        )

    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
'''.lstrip()


def _run_subprocess(
    cmd: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    timeout_s: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT after {timeout_s}s\n")
            return 124


def _run_upstream_case(
    case_name: str,
    case: Dict[str, Any],
    args: argparse.Namespace,
    runner: Path,
) -> Dict[str, Any]:
    case_root = ROOT / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    output = case_root / "result.json"
    offload_path = case_root / "offload"
    prefetch_flag = "unset"
    if case["prefetch_flag"] is True:
        prefetch_flag = "true"
    elif case["prefetch_flag"] is False:
        prefetch_flag = "false"
    cmd = [
        str(PYTHON),
        str(runner),
        "--model",
        str(UPSTREAM_MODEL),
        "--offload-path",
        str(offload_path),
        "--output",
        str(output),
        "--device-memory-ratio",
        str(args.device_memory_ratio),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--prefetch-flag",
        prefetch_flag,
    ]
    if args.dry_run:
        return {"case": case_name, "kind": "upstream", "command": cmd, "dry_run": True}
    returncode = _run_subprocess(
        cmd,
        cwd=UPSTREAM_REPO,
        env=_env(UPSTREAM_REPO),
        log_path=ROOT / "logs" / f"{case_name}.log",
        timeout_s=TIMEOUT_S,
    )
    loaded = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else {"missing_output": str(output)}
    )
    loaded.update({"case": case_name, "kind": "upstream", "returncode": returncode})
    return loaded


def _run_fgo_case(
    case_name: str,
    case: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    case_root = ROOT / case_name
    cmd = [
        str(PYTHON),
        "benchmarks/benchmark_qwen_offloading.py",
        "--model-path",
        str(FGO_MODEL),
        "--output-root",
        str(case_root),
        "--trace-dir",
        str(TRACE_DIR),
        "--variants",
        str(case["variant"]),
        "--traces",
        args.trace,
        "--warmup-requests",
        str(args.warmup_requests),
        "--measured-requests",
        str(args.measured_requests),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--fixed-new-tokens",
        "--max-input-length",
        str(args.max_input_length),
        "--device-memory-ratio",
        str(args.device_memory_ratio),
        "--num-threads",
        "1",
        "--prefetch-future-layers",
        "4" if case["prefetch"] else "0",
        "--prefetch-max-candidates",
        "32" if case["prefetch"] else "0",
        "--prefetch-admission-enabled",
        "--prefetch-credit-gated-enabled",
        "--prefetch-credit-count",
        "8" if case["prefetch"] else "0",
        "--prefetch-execution-mode",
        str(case["execution_mode"]),
        "--static-prefetch-default-topk",
        "4",
    ]
    if bool(case["protect"]):
        cmd.append("--prefetch-retention-protect-demand-eviction")
    if args.dry_run:
        return {"case": case_name, "kind": "fgo", "command": cmd, "dry_run": True}
    returncode = _run_subprocess(
        cmd,
        cwd=FGO_REPO,
        env=_env(FGO_REPO),
        log_path=ROOT / "logs" / f"{case_name}.log",
        timeout_s=TIMEOUT_S,
    )
    raw_path = case_root / "raw" / f"{args.trace}__{case['variant']}.json"
    loaded = (
        json.loads(raw_path.read_text(encoding="utf-8"))
        if raw_path.exists()
        else {"missing_raw": str(raw_path)}
    )
    loaded.update({"case": case_name, "kind": "fgo", "returncode": returncode})
    shutil.rmtree(case_root / "offload", ignore_errors=True)
    return loaded


def _summarize(results: List[Dict[str, Any]]) -> None:
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": _now(),
        "root": str(ROOT),
        "fgo_repo": str(FGO_REPO),
        "upstream_repo": str(UPSTREAM_REPO),
        "fgo_model": str(FGO_MODEL),
        "upstream_model": str(UPSTREAM_MODEL),
        "results": results,
    }
    (analysis / "original_workflow_parity_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Original Workflow Parity V1",
        "",
        "This run checks workflow parity, not paper-level performance.",
        "",
        "| case | kind | status | tok/s | prefetch calls | key observation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        case = result.get("case", "")
        kind = result.get("kind", "")
        if result.get("dry_run"):
            lines.append(f"| {case} | {kind} | dry-run | 0.000 | 0 | command only |")
            continue
        if kind == "upstream":
            status = "ok" if result.get("success") else f"failed:{result.get('error_type', 'unknown')}"
            tok_s = float(result.get("generated_tokens_per_second") or 0.0)
            calls = int(result.get("prefetch_counters", {}).get("prefetch_experts_calls", 0))
            observation = "upstream forward called prefetcher" if calls else "no upstream prefetch call observed"
        else:
            aggregate = result.get("aggregate", {})
            status = "ok" if int(result.get("returncode", 1)) == 0 and not result.get("missing_raw") else "failed"
            tok_s = float(aggregate.get("generated_tokens_per_second") or 0.0)
            calls = int(aggregate.get("prefetch_admitted_count_total", 0))
            push = int(aggregate.get("prefetch_runtime_queue_push_count_total", 0))
            hit = int(aggregate.get("dispatcher_prefetch_resident_hit_count_total", 0))
            observation = f"admitted={calls}, queue_push={push}, resident_hit={hit}"
        lines.append(f"| {case} | {kind} | {status} | {tok_s:.3f} | {calls} | {observation} |")
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- Upstream Qwen/DeepSeek runs with zero prefetch calls are workflow evidence, not negative performance evidence.",
            "- fgo runs with miss reduction but zero queue push should be treated as retention, not H2D prefetch.",
        ]
    )
    (analysis / "original_workflow_parity_v1.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run original MoE-Infinity workflow parity checks.")
    parser.add_argument("--cases", nargs="+", choices=list(CASES))
    parser.add_argument("--trace", default=os.environ.get("ORIGINAL_WORKFLOW_TRACE", "mixed"))
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=int(os.environ.get("ORIGINAL_WORKFLOW_WARMUP_REQUESTS", "1")),
    )
    parser.add_argument(
        "--measured-requests",
        type=int,
        default=int(os.environ.get("ORIGINAL_WORKFLOW_MEASURED_REQUESTS", "4")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("ORIGINAL_WORKFLOW_MAX_NEW_TOKENS", "8")),
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=int(os.environ.get("ORIGINAL_WORKFLOW_MAX_INPUT_LENGTH", "128")),
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=float(os.environ.get("ORIGINAL_WORKFLOW_DEVICE_MEMORY_RATIO", "0.60")),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "upstream_prefetch_call_probe.py"
    _write_upstream_runner(runner)
    results: List[Dict[str, Any]] = []
    for name in _selected_cases(args):
        case = CASES[name]
        if case["kind"] == "upstream":
            results.append(_run_upstream_case(name, case, args, runner))
        else:
            results.append(_run_fgo_case(name, case, args))
    _summarize(results)
    print(ROOT / "analysis" / "original_workflow_parity_v1.md")


if __name__ == "__main__":
    main()
