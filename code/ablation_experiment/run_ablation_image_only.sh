#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/evaluate_arrangement_ablation_vllm.py"
MODEL_PATH="MODEL PATH HERE"
JSONL_DIR="../../datasets/jsonl"
COHERENCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${COHERENCE_ROOT}/results/ablation_image_only}"

TP_SIZE="${TP_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
MAX_TOKENS="${MAX_TOKENS:-12120}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_PREDICTION_RETRIES="${MAX_PREDICTION_RETRIES:-0}"

BENCH_FILES=(
  "${JSONL_DIR}/cooking.jsonl"
  "${JSONL_DIR}/science.jsonl"
  "${JSONL_DIR}/storybird.jsonl"
  "${JSONL_DIR}/wikihow.jsonl"
)

mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found."
  exit 1
fi

for bench in "${BENCH_FILES[@]}"; do
  if [[ ! -f "${bench}" ]]; then
    echo "Benchmark file not found."
    exit 1
  fi
done

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found."
  echo "Please set MODEL_PATH explicitly."
  exit 1
fi

echo "=========================================="
echo "Ablation Experiment: image_only"
echo "TP=${TP_SIZE}"
echo "Model Config: [REDACTED]"
echo "=========================================="

for bench in "${BENCH_FILES[@]}"; do
  bench_base="$(basename "${bench}" .jsonl)"
  output_file="${OUTPUT_DIR}/${bench_base}_ablation_image_only.jsonl"

  echo "[RUN][image_only] benchmark=${bench_base}"

  python "${PY_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --benchmark_file "${bench}" \
    --output_file "${output_file}" \
    --ablation_mode image_only \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prediction_retries "${MAX_PREDICTION_RETRIES}" \
    --tensor_parallel_size "${TP_SIZE}"
done

echo "[DONE] image_only ablation completed"
