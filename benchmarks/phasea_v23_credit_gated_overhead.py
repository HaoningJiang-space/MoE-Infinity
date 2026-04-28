from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(
    os.environ.get(
        "V23_ROOT",
        "/data/ziheng/moe_infinity_fgo_runs/phasea_v23_credit_gated_overhead",
    )
)
REPO = Path(os.environ.get("V23_REPO", "/data/ziheng/projects/moe_infinity_fgo"))
PYTHON = Path(
    os.environ.get("V23_PYTHON", "/home/ziheng/miniconda3/envs/mxmoe/bin/python")
)
MODEL = Path(
    os.environ.get("V23_MODEL", "/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat")
)
TRACE_DIR = Path(os.environ.get("V23_TRACE_DIR", str(REPO / "benchmarks/traces/qwen")))
CUDA_VISIBLE_DEVICES = os.environ.get("V23_CUDA_VISIBLE_DEVICES", "1")
TRACE_NAME = os.environ.get("V23_TRACE_NAME", "mixed")
DEVICE_MEMORY_RATIO = os.environ.get("V23_DEVICE_MEMORY_RATIO", "0.30")
WARMUP_REQUESTS = os.environ.get("V23_WARMUP_REQUESTS", "2")
MEASURED_REQUESTS = os.environ.get("V23_MEASURED_REQUESTS", "32")
MAX_NEW_TOKENS = os.environ.get("V23_MAX_NEW_TOKENS", "16")
MAX_INPUT_LENGTH = os.environ.get("V23_MAX_INPUT_LENGTH", "128")
PHASEA_EVENTS = os.environ.get("V23_PHASEA_EVENTS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


CASES: Dict[str, Dict[str, Any]] = {
    "on_demand": {
        "variant": "on_demand",
        "future_layers": 0,
        "max_candidates": 0,
        "admission": False,
        "admission_max_per_plan": -1,
        "credit_gated": False,
        "credit_count": -1,
    },
    "cap0_generate_then_drop": {
        "variant": "history_reuse_local_backbone",
        "future_layers": 4,
        "max_candidates": 32,
        "admission": True,
        "admission_max_per_plan": 0,
        "credit_gated": False,
        "credit_count": -1,
    },
    "cap0_skip_generation": {
        "variant": "history_reuse_local_backbone",
        "future_layers": 4,
        "max_candidates": 32,
        "admission": True,
        "admission_max_per_plan": -1,
        "credit_gated": True,
        "credit_count": 0,
    },
    "cap8_generate_then_drop": {
        "variant": "history_reuse_local_backbone",
        "future_layers": 4,
        "max_candidates": 32,
        "admission": True,
        "admission_max_per_plan": 8,
        "credit_gated": False,
        "credit_count": -1,
    },
    "cap8_credit_gated_generation": {
        "variant": "history_reuse_local_backbone",
        "future_layers": 4,
        "max_candidates": 32,
        "admission": True,
        "admission_max_per_plan": -1,
        "credit_gated": True,
        "credit_count": 8,
    },
}


def _selected_case_names() -> List[str]:
    requested = os.environ.get("V23_CASES")
    if not requested:
        return list(CASES)
    names = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise ValueError(f"Unknown V23_CASES entries: {unknown}")
    return names


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _base_cmd(case_root: Path, case: Dict[str, Any]) -> List[str]:
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
        WARMUP_REQUESTS,
        "--measured-requests",
        MEASURED_REQUESTS,
        "--max-new-tokens",
        MAX_NEW_TOKENS,
        "--fixed-new-tokens",
        "--max-input-length",
        MAX_INPUT_LENGTH,
        "--device-memory-ratio",
        DEVICE_MEMORY_RATIO,
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
        str(case["admission_max_per_plan"]),
        "--prefetch-credit-count",
        str(case["credit_count"]),
        "--historical-reuse-match-topk",
        "4",
        "--historical-reuse-match-min-required",
        "2",
        "--historical-reuse-consensus-min-votes",
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
    if PHASEA_EVENTS:
        cmd.extend(
            [
                "--phasea-events",
                "--phasea-max-ranked-candidates",
                "32",
                "--phasea-analysis-future-layers",
                "0",
                "--phasea-analysis-max-ranked-candidates",
                "128",
            ]
        )
    return cmd


def _run_case(name: str) -> int:
    case = CASES[name]
    case_root = ROOT / name
    log_path = ROOT / "logs" / f"{name}.log"
    case_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    _append(ROOT / "driver.log", f"[{started}] start {name}")
    env = os.environ.copy()
    env.update(
        {
            "PATH": "/home/ziheng/miniconda3/envs/mxmoe/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": str(REPO),
            "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        }
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            _base_cmd(case_root, case),
            cwd=REPO,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    subprocess.run(["rm", "-rf", str(case_root / "offload")], check=False)
    finished = _now()
    with (ROOT / "case_status.tsv").open("a", encoding="utf-8") as handle:
        handle.write(f"{name}\t{result.returncode}\t{started}\t{finished}\n")
    status = "done" if result.returncode == 0 else f"failed({result.returncode})"
    _append(ROOT / "driver.log", f"[{finished}] {status} {name}")
    return int(result.returncode)


def _load_case_result(name: str) -> Dict[str, Any]:
    variant = str(CASES[name]["variant"])
    raw_path = ROOT / name / "raw" / f"{TRACE_NAME}__{variant}.json"
    if not raw_path.exists():
        return {"case": name, "missing": True}
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    return {
        "case": name,
        "failed": bool(payload.get("failed", False)),
        "variant": variant,
        "config": payload.get("config", {}),
        "aggregate": payload.get("aggregate", {}),
        "failure": payload.get("failure"),
        "raw_json": str(raw_path),
    }


def _write_summary(case_names: List[str]) -> None:
    results = [_load_case_result(name) for name in case_names]
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(ROOT),
        "trace": TRACE_NAME,
        "device_memory_ratio": float(DEVICE_MEMORY_RATIO),
        "phasea_events": bool(PHASEA_EVENTS),
        "cases": results,
    }
    (analysis_dir / "v23_credit_gated_overhead_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# V23 Credit-Gated Overhead Summary",
        "",
        f"- Trace: `{TRACE_NAME}`",
        f"- Device memory ratio: `{DEVICE_MEMORY_RATIO}`",
        f"- Phase-A events: `{str(PHASEA_EVENTS).lower()}`",
        "",
        "| Case | status | tok/s | mean ms/token | p95 ms/token | library queries | candidates | admitted | drops | credit skip | credit issued | credit materialized | no-victim | all-locked | pending stall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if result.get("missing"):
            lines.append(
                f"| {result['case']} | missing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |"
            )
            continue
        aggregate = result.get("aggregate", {})
        status = "failed" if result.get("failed") else "complete"
        lines.append(
            "| {case} | {status} | {tps:.3f} | {mean_ms:.2f} | {p95_ms:.2f} | {queries} | {candidates} | {admitted} | {drops} | {credit_skip} | {credit_issued} | {credit_materialized} | {no_victim} | {all_locked} | {pending_stall} |".format(
                case=result["case"],
                status=status,
                tps=float(aggregate.get("generated_tokens_per_second", 0.0)),
                mean_ms=float(
                    aggregate.get("latency_per_generated_token_mean_ms", 0.0)
                ),
                p95_ms=float(
                    aggregate.get("latency_per_generated_token_p95_ms", 0.0)
                ),
                queries=int(aggregate.get("library_query_count_total", 0)),
                candidates=int(aggregate.get("prefetch_candidate_count_total", 0)),
                admitted=int(aggregate.get("prefetch_admitted_count_total", 0)),
                drops=int(aggregate.get("prefetch_drop_count_total", 0)),
                credit_skip=int(
                    aggregate.get("prefetch_credit_skip_count_total", 0)
                ),
                credit_issued=int(
                    aggregate.get("prefetch_credit_issued_total", 0)
                ),
                credit_materialized=int(
                    aggregate.get("prefetch_credit_materialized_count_total", 0)
                ),
                no_victim=int(
                    aggregate.get("dispatcher_no_victim_wait_count_total", 0)
                ),
                all_locked=int(
                    aggregate.get("dispatcher_all_locked_event_count_total", 0)
                ),
                pending_stall=int(
                    aggregate.get("dispatcher_pending_stall_count_total", 0)
                ),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation target:",
            "",
            "- `cap0_skip_generation` isolates the control-plane tax of dropped speculation.",
            "- `cap8_credit_gated_generation` tests whether upstream credits reduce candidate materialization without losing progress.",
            "- This is an overhead-decomposition run, not a final performance sweep.",
            "",
        ]
    )
    (analysis_dir / "v23_credit_gated_overhead_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    if ROOT.exists():
        subprocess.run(["rm", "-rf", str(ROOT)], check=False)
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "case_status.tsv").write_text(
        "case\texit_code\tstarted_utc\tfinished_utc\n",
        encoding="utf-8",
    )
    case_names = _selected_case_names()
    for name in case_names:
        _run_case(name)
    _write_summary(case_names)
    _append(ROOT / "driver.log", f"[{_now()}] v23 analysis complete")


if __name__ == "__main__":
    main()
