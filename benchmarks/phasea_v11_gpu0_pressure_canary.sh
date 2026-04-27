#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ziheng/moe_infinity_fgo_runs/phasea_v11_gpu0_pressure_canary"
REPO="/data/ziheng/projects/moe_infinity_fgo"
PYTHON="/home/ziheng/miniconda3/envs/mxmoe/bin/python"
MODEL="/data/ziheng/models/Qwen1.5-MoE-A2.7B-Chat"
TRACE_DIR="$REPO/benchmarks/traces/qwen"

rm -rf "$ROOT"
mkdir -p "$ROOT/analysis"

cd "$REPO"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=0

STATUS_PATH="$ROOT/case_status.tsv"
printf "mode\ttrace\tvariant\texit_code\tstarted_utc\tfinished_utc\n" >"$STATUS_PATH"

run_case() {
  local mode="$1"
  local future_layers="$2"
  local max_candidates="$3"
  local variant="$4"
  local trace_name="mixed"
  local case_root="$ROOT/$mode"
  local log_path="$case_root/logs/${trace_name}__${variant}.log"
  local started_utc
  local finished_utc
  local exit_code
  mkdir -p "$case_root/logs" "$case_root/analysis"
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$started_utc] start ${mode} ${trace_name} ${variant}" | tee -a "$case_root/driver.log" "$ROOT/driver.log"
  set +e
  "$PYTHON" benchmarks/benchmark_qwen_offloading.py \
    --model-path "$MODEL" \
    --output-root "$case_root" \
    --trace-dir "$TRACE_DIR" \
    --variants "$variant" \
    --traces "$trace_name" \
    --warmup-requests 1 \
    --measured-requests 8 \
    --max-new-tokens 16 \
    --fixed-new-tokens \
    --max-input-length 128 \
    --device-memory-ratio 0.30 \
    --num-threads 1 \
    --library-capacity 32 \
    --library-metric cosine \
    --backbone-topk 8 \
    --prefetch-future-layers "$future_layers" \
    --prefetch-max-candidates "$max_candidates" \
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
  exit_code=$?
  set -e
  rm -rf "$case_root/offload"
  finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$mode" "$trace_name" "$variant" "$exit_code" "$started_utc" "$finished_utc" >>"$STATUS_PATH"
  if [[ "$exit_code" == "0" ]]; then
    echo "[$finished_utc] done ${mode} ${trace_name} ${variant}" | tee -a "$case_root/driver.log" "$ROOT/driver.log"
  else
    echo "[$finished_utc] failed(${exit_code}) ${mode} ${trace_name} ${variant}" | tee -a "$case_root/driver.log" "$ROOT/driver.log"
  fi
}

run_group() {
  local mode="$1"
  local future_layers="$2"
  local max_candidates="$3"
  run_case "$mode" "$future_layers" "$max_candidates" on_demand
  run_case "$mode" "$future_layers" "$max_candidates" history_reuse_consensus_backbone
  run_case "$mode" "$future_layers" "$max_candidates" history_reuse_local_backbone
}

run_group conservative 2 16
run_group aggressive 4 32

"$PYTHON" benchmarks/analyze_pressure_canary.py \
  --benchmark-root "$ROOT" \
  >"$ROOT/analysis/analyze_pressure_canary.log" 2>&1 || true

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] canary analysis complete" | tee -a "$ROOT/driver.log"
