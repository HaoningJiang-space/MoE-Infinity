#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ziheng/moe_infinity_fgo_runs/phasea_v10b_runtime_fixedlen_qwen_gpu0_strong_pressure"
REPO="/data/ziheng/projects/moe_infinity_fgo"
PYTHON="/home/ziheng/miniconda3/envs/mxmoe/bin/python"
MODEL="/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat"
TRACE_DIR="$REPO/benchmarks/traces/qwen"

rm -rf "$ROOT"
mkdir -p "$ROOT"

cd "$REPO"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

run_case() {
  local ratio_label="$1"
  local ratio="$2"
  local trace_name="$3"
  local variant="$4"
  local case_root="$ROOT/$ratio_label"
  local log_path="$case_root/logs/${trace_name}__${variant}.log"
  mkdir -p "$case_root/logs" "$case_root/analysis"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${ratio_label} ${trace_name} ${variant}" | tee -a "$case_root/driver.log" "$ROOT/driver.log"
  "$PYTHON" benchmarks/benchmark_qwen_offloading.py \
    --model-path "$MODEL" \
    --output-root "$case_root" \
    --trace-dir "$TRACE_DIR" \
    --variants "$variant" \
    --traces "$trace_name" \
    --warmup-requests 2 \
    --measured-requests 32 \
    --max-new-tokens 16 \
    --fixed-new-tokens \
    --max-input-length 128 \
    --device-memory-ratio "$ratio" \
    --num-threads 1 \
    --library-capacity 32 \
    --library-metric cosine \
    --backbone-topk 8 \
    --prefetch-future-layers 4 \
    --prefetch-max-candidates 32 \
    --historical-reuse-match-topk 4 \
    --historical-reuse-match-min-required 2 \
    --historical-reuse-consensus-min-votes 2 \
    --local-continuation-library-capacity 4096 \
    --local-continuation-key-layers 4 \
    --local-continuation-future-layers 4 \
    --local-continuation-match-topk 4 \
    --local-continuation-match-min-required 2 \
    --phasea-events \
    --phasea-max-ranked-candidates 32 \
    --phasea-analysis-future-layers 0 \
    --phasea-analysis-max-ranked-candidates 128 \
    >"$log_path" 2>&1
  rm -rf "$case_root/offload"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done ${ratio_label} ${trace_name} ${variant}" | tee -a "$case_root/driver.log" "$ROOT/driver.log"
}

run_ratio() {
  local ratio_label="$1"
  local ratio="$2"
  local ratio_root="$ROOT/$ratio_label"
  mkdir -p "$ratio_root/logs" "$ratio_root/analysis"
  for trace_name in mixed recurrence_heavy stationary; do
    run_case "$ratio_label" "$ratio" "$trace_name" on_demand
    run_case "$ratio_label" "$ratio" "$trace_name" history_reuse_consensus_backbone
    run_case "$ratio_label" "$ratio" "$trace_name" history_reuse_local_backbone
  done
  "$PYTHON" benchmarks/analyze_phasea_observations.py \
    --benchmark-root "$ratio_root" \
    --output-root "$ratio_root/analysis" \
    --exclude-warmup-events \
    >"$ratio_root/analysis/analyze.log" 2>&1
  "$PYTHON" benchmarks/summarize_phasea_decisions.py \
    --benchmark-root "$ratio_root" \
    --output-root "$ratio_root/analysis" \
    --exclude-warmup-events \
    --budgets 32 \
    --skip-horizon-analysis \
    >"$ratio_root/analysis/decision_summary.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] analysis complete ${ratio_label}" | tee -a "$ratio_root/driver.log" "$ROOT/driver.log"
}

run_ratio ratio_030 0.30
run_ratio ratio_025 0.25
