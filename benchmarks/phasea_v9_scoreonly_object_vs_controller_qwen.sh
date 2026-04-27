#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ziheng/moe_infinity_fgo_runs/phasea_v9_scoreonly_object_vs_controller_qwen"
REPO="/data/ziheng/projects/moe_infinity_fgo"
PYTHON="/home/ziheng/miniconda3/envs/mxmoe/bin/python"
MODEL="/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat"
TRACE_DIR="$REPO/benchmarks/traces/qwen"

rm -rf "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/analysis"

cd "$REPO"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

run_case() {
  local trace_name="$1"
  local variant="$2"
  local log_path="$ROOT/logs/${trace_name}__${variant}.log"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start ${trace_name} ${variant}" | tee -a "$ROOT/driver.log"
  "$PYTHON" benchmarks/benchmark_qwen_offloading.py \
    --model-path "$MODEL" \
    --output-root "$ROOT" \
    --trace-dir "$TRACE_DIR" \
    --variants "$variant" \
    --traces "$trace_name" \
    --warmup-requests 0 \
    --measured-requests 32 \
    --max-new-tokens 16 \
    --fixed-new-tokens \
    --max-input-length 128 \
    --device-memory-ratio 0.6 \
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
    --policy-score-only \
    --phasea-events \
    --phasea-max-ranked-candidates 32 \
    --phasea-analysis-future-layers 0 \
    --phasea-analysis-max-ranked-candidates 128 \
    >"$log_path" 2>&1
  rm -rf "$ROOT/offload"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done ${trace_name} ${variant}" | tee -a "$ROOT/driver.log"
}

for trace_name in mixed recurrence_heavy stationary; do
  run_case "$trace_name" history_reuse_backbone
  run_case "$trace_name" history_reuse_topk_backbone
  run_case "$trace_name" history_reuse_consensus_backbone
  run_case "$trace_name" history_reuse_consensus_backbone_retrieval
  run_case "$trace_name" history_reuse_local_backbone
done

"$PYTHON" benchmarks/analyze_phasea_observations.py \
  --benchmark-root "$ROOT" \
  --output-root "$ROOT/analysis" \
  >"$ROOT/analysis/analyze.log" 2>&1

"$PYTHON" benchmarks/summarize_phasea_decisions.py \
  --benchmark-root "$ROOT" \
  --output-root "$ROOT/analysis" \
  --budgets 32 \
  --skip-horizon-analysis \
  >"$ROOT/analysis/decision_summary.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] analysis complete" | tee -a "$ROOT/driver.log"
