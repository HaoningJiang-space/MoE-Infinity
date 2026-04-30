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
        "PVRT_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/pressure_validated_retention_transfer_v1",
    )
)
REPO = Path(os.environ.get("PVRT_REPO", "/data/ziheng/projects/moe_infinity_fgo"))
PYTHON = Path(os.environ.get("PVRT_PYTHON", "/home/ziheng/miniconda3/envs/mxmoe/bin/python"))
MODEL = Path(os.environ.get("PVRT_MODEL", "/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat"))
TRACE_DIR = Path(os.environ.get("PVRT_TRACE_DIR", str(REPO / "benchmarks/traces/qwen")))
CUDA_VISIBLE_DEVICES = os.environ.get("PVRT_CUDA_VISIBLE_DEVICES", "0")
TIMEOUT_S = int(os.environ.get("PVRT_TIMEOUT_S", "1800"))


CASES: Dict[str, Dict[str, Any]] = {
    "baseline_pre": {
        "variant": "on_demand",
        "prefetch": False,
        "mode": "disabled",
        "protect": False,
    },
    "static_replace_only_no_protect": {
        "variant": "static_hot_prefetch",
        "prefetch": True,
        "mode": "replace_only",
        "protect": False,
    },
    "static_replace_only_with_protect": {
        "variant": "static_hot_prefetch",
        "prefetch": True,
        "mode": "replace_only",
        "protect": True,
    },
    "static_enqueue_only": {
        "variant": "static_hot_prefetch",
        "prefetch": True,
        "mode": "enqueue_only",
        "protect": False,
    },
    "static_replace_enqueue_with_protect": {
        "variant": "static_hot_prefetch",
        "prefetch": True,
        "mode": "replace_and_enqueue",
        "protect": True,
    },
    "local_replace_enqueue_with_protect": {
        "variant": "history_reuse_local_backbone",
        "prefetch": True,
        "mode": "replace_and_enqueue",
        "protect": True,
    },
    "baseline_post": {
        "variant": "on_demand",
        "prefetch": False,
        "mode": "disabled",
        "protect": False,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _selected_cases(args: argparse.Namespace) -> List[str]:
    if args.cases:
        names = args.cases
    else:
        names = [item.strip() for item in os.environ.get("PVRT_CASES", "").split(",") if item.strip()]
    if not names:
        names = list(CASES)
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown cases: {unknown}")
    return names


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
            "PYTHONPATH": str(REPO),
            "PATH": "/home/ziheng/miniconda3/envs/mxmoe/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin",
            "TMPDIR": "/data/ziheng/tmp",
            "TEMP": "/data/ziheng/tmp",
            "TMP": "/data/ziheng/tmp",
        }
    )
    return env


def _cmd(case_root: Path, case: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    prefetch = bool(case["prefetch"])
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
        "--library-capacity",
        "32",
        "--library-metric",
        "cosine",
        "--backbone-topk",
        "8",
        "--prefetch-future-layers",
        "4" if prefetch else "0",
        "--prefetch-max-candidates",
        "32" if prefetch else "0",
        "--prefetch-admission-demand-reserve",
        "2",
        "--prefetch-admission-locked-ratio-threshold",
        "0.8",
        "--prefetch-admission-max-under-pressure",
        "4",
        "--prefetch-admission-max-per-plan",
        "-1",
        "--prefetch-credit-count",
        "8" if prefetch else "0",
        "--prefetch-credit-zero-action",
        "update_only",
        "--prefetch-execution-mode",
        str(case["mode"]),
        "--static-prefetch-default-topk",
        "4",
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
    if args.offload_cache_template:
        cmd.extend(
            [
                "--offload-cache-template",
                args.offload_cache_template,
                "--offload-cache-mode",
                args.offload_cache_mode,
            ]
        )
    if prefetch:
        cmd.extend(["--prefetch-admission-enabled", "--prefetch-credit-gated-enabled"])
    if bool(case["protect"]):
        cmd.append("--prefetch-retention-protect-demand-eviction")
    return cmd


def _raw_path(case_root: Path, case: Dict[str, Any], trace: str) -> Path:
    return case_root / "raw" / f"{trace}__{case['variant']}.json"


def _run_case(name: str, case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    case_root = ROOT / name
    cmd = _cmd(case_root, case, args)
    if args.dry_run:
        return {"case": name, "dry_run": True, "command": cmd}
    log_path = ROOT / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            completed = subprocess.run(
                cmd,
                cwd=REPO,
                env=_env(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_S,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            log_file.write(f"\nTIMEOUT after {TIMEOUT_S}s\n")
            returncode = 124
    raw_path = _raw_path(case_root, case, args.trace)
    loaded = (
        json.loads(raw_path.read_text(encoding="utf-8"))
        if raw_path.exists()
        else {"missing_raw": str(raw_path)}
    )
    loaded.update(
        {
            "case": name,
            "returncode": returncode,
            "execution_mode": case["mode"],
            "retention_protect": bool(case["protect"]),
        }
    )
    shutil.rmtree(case_root / "offload", ignore_errors=True)
    return loaded


def _aggregate(result: Dict[str, Any]) -> Dict[str, Any]:
    aggregate = result.get("aggregate", {})
    return {
        "case": result.get("case"),
        "variant": result.get("variant"),
        "returncode": result.get("returncode"),
        "tok_s": aggregate.get("generated_tokens_per_second", 0.0),
        "ms_per_token": aggregate.get("latency_per_generated_token_mean_ms", 0.0),
        "miss": aggregate.get("dispatcher_cache_miss_fetch_count_total", 0),
        "evict": aggregate.get("dispatcher_eviction_count_total", 0),
        "all_locked": aggregate.get("dispatcher_all_locked_event_count_total", 0),
        "no_victim": aggregate.get("dispatcher_no_victim_wait_count_total", 0),
        "pending_stall": aggregate.get("dispatcher_pending_stall_count_total", 0),
        "candidates": aggregate.get("prefetch_candidate_count_total", 0),
        "admitted": aggregate.get("prefetch_admitted_count_total", 0),
        "candidate_transfer_opportunity": aggregate.get(
            "prefetch_candidate_transfer_opportunity_count_total",
            0,
        ),
        "admitted_transfer_opportunity": aggregate.get(
            "prefetch_admitted_transfer_opportunity_count_total",
            0,
        ),
        "queue_push": aggregate.get("prefetch_runtime_queue_push_count_total", 0),
        "complete": aggregate.get("prefetch_runtime_complete_count_total", 0),
        "resident_hit": aggregate.get("dispatcher_prefetch_resident_hit_count_total", 0),
        "late_miss": aggregate.get("dispatcher_late_prefetch_demand_miss_count_total", 0),
        "execution_mode": result.get("execution_mode"),
        "retention_protect": result.get("retention_protect"),
    }


def _pressure_label(row: Dict[str, Any]) -> str:
    if row["pending_stall"] or row["no_victim"] or row["all_locked"]:
        return "progress-pressure"
    if row["evict"] > 0:
        return "eviction-pressure"
    if row["candidate_transfer_opportunity"] > 0 or row["queue_push"] > 0:
        return "transfer-diagnostic"
    return "control-or-weak-pressure"


def _summarize(results: List[Dict[str, Any]]) -> None:
    rows = [_aggregate(result) for result in results]
    baseline_values = [
        row["tok_s"]
        for row in rows
        if row["case"] in ("baseline_pre", "baseline_post") and row["tok_s"]
    ]
    baseline_mean = sum(baseline_values) / len(baseline_values) if baseline_values else None
    for row in rows:
        row["pressure_label"] = _pressure_label(row)
        row["relative_to_bracket_baseline"] = (
            row["tok_s"] / baseline_mean if baseline_mean and row["tok_s"] else None
        )
        opp = row["admitted_transfer_opportunity"]
        row["resident_hit_per_admitted_opportunity"] = row["resident_hit"] / opp if opp else 0.0

    analysis = ROOT / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": _now(),
        "root": str(ROOT),
        "baseline_mean_tok_s": baseline_mean,
        "rows": rows,
        "raw_results": results,
    }
    (analysis / "pressure_validated_retention_transfer_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Pressure Validated Retention/Transfer V1",
        "",
        f"Bracket baseline mean tok/s: {baseline_mean:.3f}" if baseline_mean else "Bracket baseline mean tok/s: n/a",
        "",
        "| case | mode | protect | pressure label | tok/s | rel | miss | evict | opp | push | complete | resident hit | hit/opp |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rel = row["relative_to_bracket_baseline"]
        rel_text = f"{rel:.3f}" if rel is not None else "n/a"
        lines.append(
            "| {case} | {mode} | {protect} | {label} | {tok:.3f} | {rel} | {miss} | {evict} | {opp} | {push} | {complete} | {hit} | {hit_opp:.3f} |".format(
                case=row["case"],
                mode=row["execution_mode"],
                protect=str(bool(row["retention_protect"])).lower(),
                label=row["pressure_label"],
                tok=float(row["tok_s"] or 0.0),
                rel=rel_text,
                miss=int(row["miss"] or 0),
                evict=int(row["evict"] or 0),
                opp=int(row["admitted_transfer_opportunity"] or 0),
                push=int(row["queue_push"] or 0),
                complete=int(row["complete"] or 0),
                hit=int(row["resident_hit"] or 0),
                hit_opp=float(row["resident_hit_per_admitted_opportunity"] or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "Validity rules:",
            "",
            "- `evict == 0` means this is not a full cache-eviction pressure point; treat it as diagnostic/control.",
            "- `queue_push == 0` and miss reduction indicates retention/protection, not H2D prefetch.",
            "- Low `resident_hit / admitted_opportunity` indicates prefetch lifecycle/deadline failure.",
        ]
    )
    (analysis / "pressure_validated_retention_transfer_v1.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pressure-validated retention/transfer ablation.")
    parser.add_argument("--cases", nargs="+", choices=list(CASES))
    parser.add_argument("--trace", default=os.environ.get("PVRT_TRACE", "mixed"))
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=int(os.environ.get("PVRT_WARMUP_REQUESTS", "2")),
    )
    parser.add_argument(
        "--measured-requests",
        type=int,
        default=int(os.environ.get("PVRT_MEASURED_REQUESTS", "16")),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("PVRT_MAX_NEW_TOKENS", "16")),
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=int(os.environ.get("PVRT_MAX_INPUT_LENGTH", "128")),
    )
    parser.add_argument(
        "--device-memory-ratio",
        type=float,
        default=float(os.environ.get("PVRT_DEVICE_MEMORY_RATIO", "0.30")),
    )
    parser.add_argument("--offload-cache-template", default=os.environ.get("PVRT_OFFLOAD_CACHE_TEMPLATE", ""))
    parser.add_argument(
        "--offload-cache-mode",
        choices=("fresh", "shared"),
        default=os.environ.get("PVRT_OFFLOAD_CACHE_MODE", "fresh"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    results = [_run_case(name, CASES[name], args) for name in _selected_cases(args)]
    _summarize(results)
    print(ROOT / "analysis" / "pressure_validated_retention_transfer_v1.md")


if __name__ == "__main__":
    main()
