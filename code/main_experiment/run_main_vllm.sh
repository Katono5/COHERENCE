#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/evaluate_arrangement_vllm.py"
MODEL_PATH="${MODEL_PATH:-<MODEL_PATH>}"
JSONL_DIR="../../datasets/jsonl"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/results}"

TP_SIZE="${TP_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_PREDICTION_RETRIES="${MAX_PREDICTION_RETRIES:-10}"

BENCH_FILES=(
  "${JSONL_DIR}/cooking.jsonl"
  "${JSONL_DIR}/science.jsonl"
  "${JSONL_DIR}/storybird.jsonl"
  "${JSONL_DIR}/wikihow.jsonl"
)

sanitize_name() {
  echo "${1}" | sed -e 's|/|_|g' -e 's|[^[:alnum:]._-]|_|g'
}

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

if [[ "${MODEL_PATH}" == "<MODEL_PATH>" || -z "${MODEL_PATH}" ]]; then
  echo "Please set MODEL_PATH before running."
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model directory not found."
  exit 1
fi

model_name="$(basename "${MODEL_PATH}")"
model_tag="$(sanitize_name "${model_name}")"
OUTPUT_DIR="${OUTPUT_ROOT}/${model_tag}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Main Experiment (vLLM)"
echo "TP: ${TP_SIZE}"
echo "Model Config: [REDACTED]"
echo "=========================================="

for bench in "${BENCH_FILES[@]}"; do
  bench_base="$(basename "${bench}" .jsonl)"
  output_file="${OUTPUT_DIR}/${bench_base}_vllm_eval.jsonl"

  echo "[RUN][main] benchmark=${bench_base}"

  python "${PY_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --benchmark_file "${bench}" \
    --output_file "${output_file}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prediction_retries "${MAX_PREDICTION_RETRIES}" \
    --tensor_parallel_size "${TP_SIZE}"
done

echo "[DONE] main experiment completed"
