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
        "SOURCE_ONLY_FAIR_MATRIX_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/source_only_fair_matrix_v1",
    )
)
FGO_REPO = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_FGO_REPO",
        "/data/ziheng/projects/moe_infinity_fgo",
    )
)
UPSTREAM_REPO = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_UPSTREAM_REPO",
        "/data/ziheng/projects/MoE-Infinity_clean_d617801",
    )
)
FGO_PYTHON = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_FGO_PYTHON",
        "/home/ziheng/miniconda3/envs/mxmoe/bin/python",
    )
)
UPSTREAM_PYTHON = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_UPSTREAM_PYTHON",
        "/data/ziheng/conda_envs/moeinf-upstream-generate/bin/python",
    )
)
MODEL = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_MODEL",
        "/data/ziheng/models/DeepSeek-V2-Lite",
    )
)
TRACE_FILE = Path(
    os.environ.get(
        "SOURCE_ONLY_FAIR_MATRIX_TRACE_FILE",
        "/data/ziheng/moe_infinity_fgo_runs/readme_mixed_workload_mirror_discover_per64_v2/mixed.jsonl",
    )
)
CUDA_VISIBLE_DEVICES = os.environ.get("SOURCE_ONLY_FAIR_MATRIX_CUDA_VISIBLE_DEVICES", "0")
TIMEOUT_S = int(os.environ.get("SOURCE_ONLY_FAIR_MATRIX_TIMEOUT_S", "3600"))


CASES: Dict[str, Dict[str, Any]] = {
    "upstream_plain": {
        "repo": "upstream",
        "prefetch_flag": None,
        "counter_hooks": False,
    },
    "upstream_counter": {
        "repo": "upstream",
        "prefetch_flag": None,
        "counter_hooks": True,
    },
    "upstream_prefetch_flag_counter": {
        "repo": "upstream",
        "prefetch_flag": True,
        "counter_hooks": True,
    },
    "fgo_plain": {
        "repo": "fgo",
        "prefetch_flag": None,
        "counter_hooks": False,
    },
    "fgo_counter": {
        "repo": "fgo",
        "prefetch_flag": None,
        "counter_hooks": True,
    },
    "fgo_prefetch_flag_counter": {
        "repo": "fgo",
        "prefetch_flag": True,
        "counter_hooks": True,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected(values: Iterable[str] | None, env_key: str, default: List[str]) -> List[str]:
    if values:
        names = list(values)
    else:
        names = [
            item.strip()
            for item in os.environ.get(env_key, "").split(",")
            if item.strip()
        ]
    return names or default


def _repo(case: Dict[str, Any]) -> Path:
    return FGO_REPO if case["repo"] == "fgo" else UPSTREAM_REPO


def _python(case: Dict[str, Any]) -> Path:
    return FGO_PYTHON if case["repo"] == "fgo" else UPSTREAM_PYTHON


def _env(repo: Path, python: Path) -> Dict[str, str]:
    env = os.environ.copy()
    extra_pythonpath = ""
    if repo == FGO_REPO:
        extra_pythonpath = os.environ.get(
            "SOURCE_ONLY_FAIR_MATRIX_FGO_PYTHONPATH_PREFIX", ""
        )
    elif repo == UPSTREAM_REPO:
        extra_pythonpath = os.environ.get(
            "SOURCE_ONLY_FAIR_MATRIX_UPSTREAM_PYTHONPATH_PREFIX", ""
        )
    pythonpath = str(repo)
    if extra_pythonpath:
        pythonpath = f"{extra_pythonpath}:{pythonpath}"
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "PYTHONPATH": pythonpath,
            "PATH": (
                f"{python.parent}:/usr/local/cuda-12.8/bin:"
                "/usr/local/bin:/usr/bin:/bin"
            ),
            "TMPDIR": "/data/ziheng/tmp",
            "TEMP": "/data/ziheng/tmp",
            "TMP": "/data/ziheng/tmp",
            "TORCH_EXTENSIONS_DIR": "/data/ziheng/torch_extensions",
        }
    )
    return env


def _write_runner(path: Path) -> None:
    path.write_text(_SOURCE_ONLY_RUNNER, encoding="utf-8")


_SOURCE_ONLY_RUNNER = r'''
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
    def __init__(self, engine, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_prefilling = None
        self.prefilling_time = None
        self.start_decoding = None
        self.decoding_time = None
        self.decoding_iterations = 0
        self.engine = engine

    def _clear_cache_counts(self) -> None:
        try:
            dispatcher = getattr(self.engine, "expert_dispatcher", None)
            if dispatcher is not None:
                dispatcher.clear_expert_cache_counts()
        except Exception:
            pass

    def put(self, value):
        if self.start_prefilling is None:
            self.start_prefilling = time.time()
            return
        if self.prefilling_time is None:
            self.prefilling_time = time.time() - self.start_prefilling
            self._clear_cache_counts()
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


def maybe_wrap_prefetch(counter_hooks: bool, counters: Dict[str, int]) -> None:
    if not counter_hooks:
        return
    try:
        import moe_infinity.memory.expert_prefetcher as expert_prefetcher_mod
    except Exception:
        return

    def wrap(method_name: str, counter_name: str) -> None:
        if not hasattr(expert_prefetcher_mod.ExpertPrefetcher, method_name):
            return
        original = getattr(expert_prefetcher_mod.ExpertPrefetcher, method_name)

        def wrapped(self, *a, **kw):
            counters[counter_name] += 1
            return original(self, *a, **kw)

        setattr(expert_prefetcher_mod.ExpertPrefetcher, method_name, wrapped)

    wrap("prefetch_experts", "prefetch_experts_calls")
    wrap("prefetch_experts_list", "prefetch_experts_list_calls")
    wrap("fetch_experts_lock_cache", "fetch_experts_lock_cache_calls")


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
    parser.add_argument("--counter-hooks", action="store_true")
    parser.add_argument("--repeat", type=int, required=True)
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
        "counter_hooks": bool(args.counter_hooks),
        "repeat": args.repeat,
        "success": False,
        "evidence_label": "source-only baseline",
        "workflow": "moe_generate",
        "prefetch_counters": counters,
    }

    total_start = time.time()
    try:
        import transformers
        from moe_infinity import MoE

        maybe_wrap_prefetch(args.counter_hooks, counters)

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
            streamer = StopWatch(getattr(model, "engine", None), tokenizer)
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


def _run_subprocess(cmd: List[str], *, cwd: Path, env: Dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
            )
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT after {TIMEOUT_S}s\n")
            return 124


def _prefetch_flag(case: Dict[str, Any]) -> str:
    if case["prefetch_flag"] is True:
        return "true"
    if case["prefetch_flag"] is False:
        return "false"
    return "unset"


def _run_case(
    *,
    case_name: str,
    case: Dict[str, Any],
    repeat: int,
    runner: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    repo = _repo(case)
    python = _python(case)
    case_root = ROOT / case_name / f"repeat_{repeat:02d}"
    offload_path = case_root / "offload"
    output = case_root / "result.json"
    case_root.mkdir(parents=True, exist_ok=True)
    if args.reset_offload:
        shutil.rmtree(offload_path, ignore_errors=True)
    cmd = [
        str(python),
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
        _prefetch_flag(case),
        "--repeat",
        str(repeat),
    ]
    if case["counter_hooks"]:
        cmd.append("--counter-hooks")
    if args.dry_run:
        return {
            "case": case_name,
            "repeat": repeat,
            "repo": case["repo"],
            "dry_run": True,
            "command": cmd,
        }
    returncode = _run_subprocess(
        cmd,
        cwd=repo,
        env=_env(repo, python),
        log_path=ROOT / "logs" / f"{case_name}__repeat_{repeat:02d}.log",
    )
    loaded = (
        json.loads(output.read_text(encoding="utf-8"))
        if output.exists()
        else {"missing_output": str(output)}
    )
    loaded.update(
        {
            "case": case_name,
            "repeat": repeat,
            "repo": case["repo"],
            "returncode": returncode,
        }
    )
    return loaded


def _prefetch_call_count(result: Dict[str, Any]) -> int:
    counters = result.get("prefetch_counters", {})
    return int(counters.get("prefetch_experts_calls", 0)) + int(
        counters.get("prefetch_experts_list_calls", 0)
    )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(results: List[Dict[str, Any]]) -> None:
    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": _now(),
        "root": str(ROOT),
        "fgo_repo": str(FGO_REPO),
        "upstream_repo": str(UPSTREAM_REPO),
        "fgo_python": str(FGO_PYTHON),
        "upstream_python": str(UPSTREAM_PYTHON),
        "fgo_pythonpath_prefix": os.environ.get(
            "SOURCE_ONLY_FAIR_MATRIX_FGO_PYTHONPATH_PREFIX", ""
        ),
        "upstream_pythonpath_prefix": os.environ.get(
            "SOURCE_ONLY_FAIR_MATRIX_UPSTREAM_PYTHONPATH_PREFIX", ""
        ),
        "model": str(MODEL),
        "trace_file": str(TRACE_FILE),
        "results": results,
    }
    (analysis / "source_only_fair_matrix_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("case", "")), []).append(result)

    lines = [
        "# Source-Only Fair Matrix V1",
        "",
        "All cases call `MoE(...); model.generate(...)`. Counter hooks only wrap",
        "prefetch methods for observation and do not manually call prefetch.",
        "",
        "| case | repo | repeats ok/total | mean TPOT ms | mean tok/s | mean wall s | mean setup s | prefetch calls | label |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case_name, items in grouped.items():
        ok = [
            item
            for item in items
            if item.get("success") and int(item.get("returncode", 1)) == 0
        ]
        label = "source-only baseline" if ok else "environment failure"
        lines.append(
            "| {case} | {repo} | {ok}/{total} | {tpot:.3f} | {tps:.3f} | {wall:.3f} | {setup:.3f} | {calls} | {label} |".format(
                case=case_name,
                repo=items[0].get("repo", ""),
                ok=len(ok),
                total=len(items),
                tpot=_mean([float(item.get("decode_tpot_ms") or 0.0) for item in ok]),
                tps=_mean([float(item.get("decode_tokens_per_second") or 0.0) for item in ok]),
                wall=_mean([float(item.get("total_wall_s") or 0.0) for item in ok]),
                setup=_mean([float(item.get("setup_s") or 0.0) for item in ok]),
                calls=sum(_prefetch_call_count(item) for item in items),
                label=label,
            )
        )

    lines.extend(
        [
            "",
            "Interpretation rules:",
            "",
            "- Valid baseline rows must be `source-only baseline` and must use `model.generate()`.",
            "- Zero prefetch calls means the target model path did not exercise activation-aware prefetch.",
            "- Rows with `counter_hooks=false` quantify plain source-only performance; rows with hooks quantify observation overhead.",
        ]
    )
    (analysis / "source_only_fair_matrix_v1.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source-only upstream/FGO fair matrix.")
    parser.add_argument("--cases", nargs="+", choices=list(CASES))
    parser.add_argument(
        "--repeats",
        type=int,
        default=int(os.environ.get("SOURCE_ONLY_FAIR_MATRIX_REPEATS", "1")),
    )
    parser.add_argument(
        "--measured-requests",
        type=int,
        default=int(os.environ.get("SOURCE_ONLY_FAIR_MATRIX_MEASURED_REQUESTS", "16")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("SOURCE_ONLY_FAIR_MATRIX_MAX_NEW_TOKENS", "16")),
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=int(os.environ.get("SOURCE_ONLY_FAIR_MATRIX_MAX_INPUT_LENGTH", "128")),
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=float(os.environ.get("SOURCE_ONLY_FAIR_MATRIX_DEVICE_MEMORY_RATIO", "0.60")),
    )
    parser.add_argument("--reset-offload", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "source_only_generate_runner.py"
    _write_runner(runner)
    cases = _selected(
        args.cases,
        "SOURCE_ONLY_FAIR_MATRIX_CASES",
        ["upstream_plain", "upstream_counter", "fgo_plain", "fgo_counter"],
    )
    results: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        for case_name in cases:
            results.append(
                _run_case(
                    case_name=case_name,
                    case=CASES[case_name],
                    repeat=repeat,
                    runner=runner,
                    args=args,
                )
            )
    _summarize(results)
    print(ROOT / "analysis" / "source_only_fair_matrix_v1.md")


if __name__ == "__main__":
    main()
