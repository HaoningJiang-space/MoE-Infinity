from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/ziheng/moe_infinity_fgo_runs/phasea_v11_gpu0_pressure_canary")
REPO = Path("/data/ziheng/projects/moe_infinity_fgo")
PYTHON = Path("/home/ziheng/miniconda3/envs/mxmoe/bin/python")
MODEL = Path("/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat")
TRACE_DIR = REPO / "benchmarks/traces/qwen"
TRACE_NAME = "mixed"
CASES = [
    ("conservative", 2, 16, "on_demand"),
    ("conservative", 2, 16, "history_reuse_consensus_backbone"),
    ("conservative", 2, 16, "history_reuse_local_backbone"),
    ("aggressive", 4, 32, "on_demand"),
    ("aggressive", 4, 32, "history_reuse_consensus_backbone"),
    ("aggressive", 4, 32, "history_reuse_local_backbone"),
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _run_case(mode: str, future_layers: int, max_candidates: int, variant: str) -> None:
    case_root = ROOT / mode
    log_path = case_root / "logs" / f"{TRACE_NAME}__{variant}.log"
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "logs").mkdir(parents=True, exist_ok=True)
    (case_root / "analysis").mkdir(parents=True, exist_ok=True)
    started = _now()
    start_line = f"[{started}] start {mode} {TRACE_NAME} {variant}"
    _append(ROOT / "driver.log", start_line)
    _append(case_root / "driver.log", start_line)
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
        variant,
        "--traces",
        TRACE_NAME,
        "--warmup-requests",
        "1",
        "--measured-requests",
        "8",
        "--max-new-tokens",
        "16",
        "--fixed-new-tokens",
        "--max-input-length",
        "128",
        "--device-memory-ratio",
        "0.30",
        "--num-threads",
        "1",
        "--library-capacity",
        "32",
        "--library-metric",
        "cosine",
        "--backbone-topk",
        "8",
        "--prefetch-future-layers",
        str(future_layers),
        "--prefetch-max-candidates",
        str(max_candidates),
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
        "--phasea-events",
        "--phasea-max-ranked-candidates",
        "32",
        "--phasea-analysis-future-layers",
        "0",
        "--phasea-analysis-max-ranked-candidates",
        "128",
    ]
    env = {
        "PATH": "/home/ziheng/miniconda3/envs/mxmoe/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": str(REPO),
        "CUDA_VISIBLE_DEVICES": "0",
    }
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            cmd,
            cwd=REPO,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    subprocess.run(["rm", "-rf", str(case_root / "offload")], check=False)
    finished = _now()
    with (ROOT / "case_status.tsv").open("a", encoding="utf-8") as handle:
        handle.write(
            f"{mode}\t{TRACE_NAME}\t{variant}\t{result.returncode}\t{started}\t{finished}\n"
        )
    status = "done" if result.returncode == 0 else f"failed({result.returncode})"
    finish_line = f"[{finished}] {status} {mode} {TRACE_NAME} {variant}"
    _append(ROOT / "driver.log", finish_line)
    _append(case_root / "driver.log", finish_line)


def main() -> None:
    if ROOT.exists():
        subprocess.run(["rm", "-rf", str(ROOT)], check=False)
    (ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (ROOT / "case_status.tsv").write_text(
        "mode\ttrace\tvariant\texit_code\tstarted_utc\tfinished_utc\n",
        encoding="utf-8",
    )
    for mode, future_layers, max_candidates, variant in CASES:
        _run_case(mode, future_layers, max_candidates, variant)
    subprocess.run(
        [
            str(PYTHON),
            "benchmarks/analyze_pressure_canary.py",
            "--benchmark-root",
            str(ROOT),
        ],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO)},
        check=False,
    )
    _append(ROOT / "driver.log", f"[{_now()}] canary analysis complete")


if __name__ == "__main__":
    main()
