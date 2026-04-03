#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COHERENCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/evaluate_arrangement_vllm.py"
MODEL_PATH=""
BENCH_DIR="${COHERENCE_ROOT}/datasets/benchmark_data/full_benchmark_7670"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/results}"

TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
MAX_TOKENS="${MAX_TOKENS:-12000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_PREDICTION_RETRIES="${MAX_PREDICTION_RETRIES:-0}"

BENCH_FILES=(
  "${BENCH_DIR}/cooking_full.reasonable.jsonl"
  "${BENCH_DIR}/science_full.reasonable.jsonl"
  "${BENCH_DIR}/storybird_full.reasonable.jsonl"
  "${BENCH_DIR}/wikihow_full.reasonable.jsonl"
)

sanitize_name() {
  echo "${1}" | sed -e 's|/|_|g' -e 's|[^[:alnum:]._-]|_|g'
}

model_name="$(basename "${MODEL_PATH}")"
model_tag="$(sanitize_name "${model_name}")"
OUTPUT_DIR="${OUTPUT_ROOT}/${model_tag}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Main Experiment (vLLM)"
echo "Model: ${MODEL_PATH}"
echo "TP: ${TP_SIZE}"
echo "DP: ${DP_SIZE}"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

for bench in "${BENCH_FILES[@]}"; do
  bench_base="$(basename "${bench}" .jsonl)"
  output_file="${OUTPUT_DIR}/${bench_base}_vllm_eval.jsonl"

  echo "[RUN][main] benchmark=${bench}"
  echo "[RUN][main] output=${output_file}"

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
    --tensor_parallel_size "${TP_SIZE}" \
    --data_parallel_size "${DP_SIZE}"
done

echo "[DONE] main experiment 全部完成"
