#!/usr/bin/env bash
set -euo pipefail

ROOT="${V36_ROOT:-/data/ziheng/moe_infinity_fgo_runs/phasea_v36_bracketed_fair_benchmark}"
REPO="${V36_REPO:-/data/ziheng/projects/moe_infinity_fgo}"
MODEL="${V36_MODEL:-/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat}"
OFFLOAD_CACHE="${V36_OFFLOAD_CACHE_TEMPLATE:-/data/ziheng/moe_infinity_fgo_runs/offload_cache/qwen15_moe_a27b_ratio030}"
PYTHON="${V36_PYTHON:-/home/ziheng/miniconda3/envs/mxmoe/bin/python}"

rm -rf "${ROOT}"
mkdir -p "${ROOT}"

cd "${REPO}"
CUDA_VISIBLE_DEVICES="${V36_CUDA_VISIBLE_DEVICES:-0}" \
PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
DEV_ROOT="${ROOT}" \
DEV_REPO="${REPO}" \
DEV_MODEL="${MODEL}" \
DEV_OFFLOAD_CACHE_TEMPLATE="${OFFLOAD_CACHE}" \
DEV_TRACE_DIR="${V36_TRACE_DIR:-${REPO}/benchmarks/traces/qwen}" \
DEV_TRACE_NAME="${V36_TRACE_NAME:-mixed}" \
DEV_DEVICE_MEMORY_RATIO="${V36_DEVICE_MEMORY_RATIO:-0.30}" \
DEV_WARMUP_REQUESTS="${V36_WARMUP_REQUESTS:-4}" \
DEV_MEASURED_REQUESTS="${V36_MEASURED_REQUESTS:-32}" \
DEV_MAX_NEW_TOKENS="${V36_MAX_NEW_TOKENS:-16}" \
DEV_REPEATS="${V36_REPEATS:-3}" \
DEV_BRACKETED_BASELINE=1 \
DEV_SHUFFLE_CASES=1 \
DEV_RANDOM_SEED="${V36_RANDOM_SEED:-36}" \
DEV_DRIFT_INVALID_THRESHOLD="${V36_DRIFT_INVALID_THRESHOLD:-0.10}" \
DEV_RESET_BETWEEN_CASES=1 \
DEV_CASES="${V36_CASES:-prefetch_enabled_no_policy,static_top4_replace_only,static_top4_enqueue_only,static_top4_replace_and_enqueue}" \
"${PYTHON}" benchmarks/phasea_dev_inprocess_static_smoke.py
