from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(
    os.environ.get(
        "ORIGINAL_UPSTREAM_BASELINE_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/original_upstream_baseline_v1",
    )
)
FGO_REPO = Path(
    os.environ.get(
        "ORIGINAL_UPSTREAM_BASELINE_FGO_REPO",
        "/data/ziheng/projects/moe_infinity_fgo",
    )
)
UPSTREAM_REPO = Path(
    os.environ.get(
        "ORIGINAL_UPSTREAM_BASELINE_UPSTREAM_REPO",
        "/data/ziheng/projects/MoE-Infinity",
    )
)
PYTHON = Path(
    os.environ.get(
        "ORIGINAL_UPSTREAM_BASELINE_PYTHON",
        "/data/ziheng/conda_envs/moeinf-upstream-generate/bin/python",
    )
)
MODEL = Path(
    os.environ.get(
        "ORIGINAL_UPSTREAM_BASELINE_MODEL",
        "/data/ziheng/models/DeepSeek-V2-Lite",
    )
)
TRACE_FILE = Path(
    os.environ.get(
        "ORIGINAL_UPSTREAM_BASELINE_TRACE_FILE",
        str(FGO_REPO / "benchmarks/traces/qwen/mixed.jsonl"),
    )
)
CUDA_VISIBLE_DEVICES = os.environ.get(
    "ORIGINAL_UPSTREAM_BASELINE_CUDA_VISIBLE_DEVICES", "0"
)
TIMEOUT_S = int(os.environ.get("ORIGINAL_UPSTREAM_BASELINE_TIMEOUT_S", "2400"))


CASES: Dict[str, Dict[str, Any]] = {
    "upstream_readme_default": {"prefetch_flag": None},
    "upstream_prefetch_true": {"prefetch_flag": True},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected(
    values: Iterable[str] | None, env_key: str, default: List[str]
) -> List[str]:
    if values:
        names = list(values)
    else:
        names = [
            item.strip()
            for item in os.environ.get(env_key, "").split(",")
            if item.strip()
        ]
    return names or default


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "PYTHONPATH": str(UPSTREAM_REPO),
            "PATH": (
                f"{PYTHON.parent}:/usr/local/cuda-12.8/bin:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
            "TMPDIR": "/data/ziheng/tmp",
            "TEMP": "/data/ziheng/tmp",
            "TMP": "/data/ziheng/tmp",
        }
    )
    return env


def _write_runner(path: Path) -> None:
    path.write_text(_UPSTREAM_SOURCE_RUNNER, encoding="utf-8")


_UPSTREAM_SOURCE_RUNNER = r'''
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, TextStreamer


class StopWatch(TextStreamer):
    """Timing-only copy of the upstream example's streamer pattern."""

    def __init__(self, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0
        self.engine = engine

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        if self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self.engine.expert_dispatcher.clear_expert_cache_counts()
            self.start_decoding = time.time()
        self.decoding_iterations += 1
        return super().put(value)

    def end(self):
        if self.decoding_time is None and self.start_decoding is not None:
            self.decoding_time = time.time() - self.start_decoding
        return super().end()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return float(values[idx])


def prompt_from_record(record: Dict[str, Any], tokenizer: Any) -> str:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    parts.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
            return "\n".join(parts)
    for key in ("prompt", "text", "input", "content"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return "Explain expert offloading for MoE inference."


def load_prompts(trace_file: Path, tokenizer: Any, measured_requests: int) -> List[str]:
    prompts: List[str] = []
    with trace_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) >= measured_requests:
                break
            line = line.strip()
            if not line:
                continue
            prompts.append(prompt_from_record(json.loads(line), tokenizer))
    if not prompts:
        prompts.append("Explain expert offloading for MoE inference.")
    return prompts


def custom_generate_kwargs(model_name: str, tokenizer: Any) -> Dict[str, Any]:
    lower = model_name.lower()
    if "switch" in lower:
        return {"decoder_start_token_id": 0}
    if "nllb" in lower:
        return {"forced_bos_token_id": 256057}
    if any(name in lower for name in ("mixtral", "arctic", "deepseek", "qwen3")):
        return {"pad_token_id": tokenizer.eos_token_id}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--offload-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-memory-ratio", type=float, required=True)
    parser.add_argument("--max-input-length", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--measured-requests", type=int, required=True)
    parser.add_argument("--prefetch-flag", choices=("unset", "true", "false"), default="unset")
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    args = parser.parse_args()

    counters = {
        "prefetch_experts_calls": 0,
        "prefetch_experts_list_calls": 0,
        "fetch_experts_lock_cache_calls": 0,
    }
    result: Dict[str, Any] = {
        "model": args.model,
        "trace_file": args.trace_file,
        "offload_path": args.offload_path,
        "device_memory_ratio": args.device_memory_ratio,
        "max_input_length": args.max_input_length,
        "max_new_tokens": args.max_new_tokens,
        "measured_requests": args.measured_requests,
        "prefetch_flag": args.prefetch_flag,
        "phase": args.phase,
        "success": False,
        "evidence_label": "source-only baseline",
        "workflow": "upstream_moe_generate",
        "prefetch_counters": counters,
    }

    total_start = time.time()
    try:
        import transformers
        import moe_infinity.memory.expert_prefetcher as expert_prefetcher_mod
        from moe_infinity import MoE

        def wrap(method_name: str, counter_name: str) -> None:
            original = getattr(expert_prefetcher_mod.ExpertPrefetcher, method_name)

            def wrapped(self, *a, **kw):
                counters[counter_name] += 1
                return original(self, *a, **kw)

            setattr(expert_prefetcher_mod.ExpertPrefetcher, method_name, wrapped)

        wrap("prefetch_experts", "prefetch_experts_calls")
        wrap("prefetch_experts_list", "prefetch_experts_list_calls")
        wrap("fetch_experts_lock_cache", "fetch_experts_lock_cache_calls")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
            use_fast=False,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = load_prompts(Path(args.trace_file), tokenizer, args.measured_requests)

        config = {
            "offload_path": args.offload_path,
            "device_memory_ratio": args.device_memory_ratio,
        }
        if args.prefetch_flag != "unset":
            config["prefetch"] = args.prefetch_flag == "true"

        setup_start = time.time()
        model = MoE(args.model, config)
        setup_s = time.time() - setup_start
        generate_kwargs = custom_generate_kwargs(args.model, tokenizer)

        records = []
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            )
            input_ids = encoded.input_ids.to("cuda:0")
            streamer = StopWatch(model.engine, tokenizer)
            torch.cuda.synchronize()
            request_start = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    streamer=streamer,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    **generate_kwargs,
                )
            torch.cuda.synchronize()
            request_s = time.time() - request_start
            generated_tokens = max(0, int(output_ids.shape[-1] - input_ids.shape[-1]))
            records.append(
                {
                    "input_tokens": int(input_ids.shape[-1]),
                    "generated_tokens": generated_tokens,
                    "prefill_s": float(streamer.prefilling_time or 0.0),
                    "decode_s": float(streamer.decoding_time or 0.0),
                    "decode_iterations": int(streamer.decoding_iterations),
                    "request_s": float(request_s),
                }
            )

        total_wall_s = time.time() - total_start
        prefill_s = sum(float(item["prefill_s"]) for item in records)
        decode_s = sum(float(item["decode_s"]) for item in records)
        request_s = [float(item["request_s"]) for item in records]
        generated_tokens = sum(int(item["generated_tokens"]) for item in records)
        decode_tokens = sum(int(item["decode_iterations"]) for item in records)
        result.update(
            {
                "success": True,
                "setup_s": setup_s,
                "total_wall_s": total_wall_s,
                "request_count": len(records),
                "generated_tokens": generated_tokens,
                "decode_tokens": decode_tokens,
                "prefill_s_total": prefill_s,
                "decode_s_total": decode_s,
                "decode_tpot_ms": (decode_s / decode_tokens * 1000.0) if decode_tokens else 0.0,
                "decode_tokens_per_second": (decode_tokens / decode_s) if decode_s else 0.0,
                "request_latency_mean_s": statistics.mean(request_s) if request_s else 0.0,
                "request_latency_p50_s": percentile(request_s, 50.0),
                "request_latency_p95_s": percentile(request_s, 95.0),
                "prefetch_counters": counters,
                "records": records,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
            }
        )
    except Exception as exc:
        result.update(
            {
                "success": False,
                "evidence_label": "environment failure",
                "total_wall_s": time.time() - total_start,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=30),
                "prefetch_counters": counters,
            }
        )

    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
'''.lstrip()


def _run_subprocess(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cmd,
                cwd=UPSTREAM_REPO,
                env=_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT after {TIMEOUT_S}s\n")
            return 124


def _run_phase(
    *,
    case_name: str,
    case: Dict[str, Any],
    phase: str,
    runner: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    case_root = ROOT / case_name
    offload_path = case_root / "offload"
    phase_root = case_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    if phase == "cold":
        shutil.rmtree(offload_path, ignore_errors=True)
    prefetch_flag = "unset"
    if case["prefetch_flag"] is True:
        prefetch_flag = "true"
    elif case["prefetch_flag"] is False:
        prefetch_flag = "false"
    output = phase_root / "result.json"
    cmd = [
        str(PYTHON),
        str(runner),
        "--model",
        str(MODEL),
        "--trace-file",
        str(TRACE_FILE),
        "--offload-path",
        str(offload_path),
        "--output",
        str(output),
        "--device-memory-ratio",
        str(args.device_memory_ratio),
        "--max-input-length",
        str(args.max_input_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--measured-requests",
        str(args.measured_requests),
        "--prefetch-flag",
        prefetch_flag,
        "--phase",
        phase,
    ]
    if args.dry_run:
        return {
            "case": case_name,
            "phase": phase,
            "dry_run": True,
            "command": cmd,
        }
    returncode = _run_subprocess(cmd, ROOT / "logs" / f"{case_name}__{phase}.log")
    loaded = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else {"missing_output": str(output)}
    )
    loaded.update({"case": case_name, "phase": phase, "returncode": returncode})
    return loaded


def _prefetch_call_count(result: Dict[str, Any]) -> int:
    counters = result.get("prefetch_counters", {})
    return int(counters.get("prefetch_experts_calls", 0)) + int(
        counters.get("prefetch_experts_list_calls", 0)
    )


def _summarize(results: List[Dict[str, Any]]) -> None:
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": _now(),
        "root": str(ROOT),
        "upstream_repo": str(UPSTREAM_REPO),
        "python": str(PYTHON),
        "model": str(MODEL),
        "trace_file": str(TRACE_FILE),
        "results": results,
    }
    (analysis / "original_upstream_baseline_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Original Upstream Baseline V1",
        "",
        "This run is source-only: it calls upstream `MoE.generate()` and does",
        "not use a custom decode loop, manual KV-cache propagation, or manual",
        "prefetch calls.",
        "",
        "| case | phase | evidence | status | decode TPOT ms | decode tok/s | wall s | setup s | requests | prefetch calls | error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        status = (
            "ok"
            if result.get("success") and int(result.get("returncode", 1)) == 0
            else f"failed:{result.get('error_type', 'unknown')}"
        )
        lines.append(
            "| {case} | {phase} | {label} | {status} | {tpot:.3f} | {tps:.3f} | {wall:.3f} | {setup:.3f} | {req} | {calls} | {error} |".format(
                case=result.get("case", ""),
                phase=result.get("phase", ""),
                label=result.get("evidence_label", ""),
                status=status,
                tpot=float(result.get("decode_tpot_ms") or 0.0),
                tps=float(result.get("decode_tokens_per_second") or 0.0),
                wall=float(result.get("total_wall_s") or 0.0),
                setup=float(result.get("setup_s") or 0.0),
                req=int(result.get("request_count") or 0),
                calls=_prefetch_call_count(result),
                error=str(result.get("error", ""))[:120].replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- `source-only baseline` is valid baseline evidence only when status is `ok`.",
            "- `environment failure` means the original source workflow failed; do not replace it with a custom decode loop for baseline tables.",
            "- If prefetch calls are zero, the target upstream model path did not exercise activation-aware prefetch.",
        ]
    )
    (analysis / "original_upstream_baseline_v1.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source-only upstream MoE-Infinity baseline."
    )
    parser.add_argument("--cases", nargs="+", choices=list(CASES))
    parser.add_argument("--phases", nargs="+", choices=("cold", "warm"))
    parser.add_argument(
        "--measured-requests",
        type=int,
        default=int(
            os.environ.get("ORIGINAL_UPSTREAM_BASELINE_MEASURED_REQUESTS", "16")
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("ORIGINAL_UPSTREAM_BASELINE_MAX_NEW_TOKENS", "16")),
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=int(
            os.environ.get("ORIGINAL_UPSTREAM_BASELINE_MAX_INPUT_LENGTH", "128")
        ),
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=float(
            os.environ.get("ORIGINAL_UPSTREAM_BASELINE_DEVICE_MEMORY_RATIO", "0.60")
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "upstream_source_generate_runner.py"
    _write_runner(runner)
    cases = _selected(args.cases, "ORIGINAL_UPSTREAM_BASELINE_CASES", list(CASES))
    phases = _selected(
        args.phases, "ORIGINAL_UPSTREAM_BASELINE_PHASES", ["cold", "warm"]
    )
    results: List[Dict[str, Any]] = []
    for case_name in cases:
        for phase in phases:
            results.append(
                _run_phase(
                    case_name=case_name,
                    case=CASES[case_name],
                    phase=phase,
                    runner=runner,
                    args=args,
                )
            )
    _summarize(results)
    print(ROOT / "analysis" / "original_upstream_baseline_v1.md")


if __name__ == "__main__":
    main()
