#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COHERENCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/evaluate_arrangement_api.py"
BENCH_DIR="${COHERENCE_ROOT}/datasets/benchmark_data/full_benchmark_7670"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/results}"

API_BASE="${API_BASE:-}"
API_KEY="${API_KEY:-}"
API_MODEL="${API_MODEL:-}"

BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_TOKENS="${MAX_TOKENS:-6000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
MAX_PREDICTION_RETRIES="${MAX_PREDICTION_RETRIES:-0}"
IMAGE_URL_MODE="${IMAGE_URL_MODE:-data_uri}"

BENCH_FILES=(
  "${BENCH_DIR}/cooking_full.reasonable.jsonl"
  "${BENCH_DIR}/science_full.reasonable.jsonl"
  "${BENCH_DIR}/storybird_full.reasonable.jsonl"
  "${BENCH_DIR}/wikihow_full.reasonable.jsonl"
)

sanitize_name() {
  echo "${1}" | sed -e 's|/|_|g' -e 's|[^[:alnum:]._-]|_|g'
}

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "找不到脚本: ${PY_SCRIPT}"
  exit 1
fi

for bench in "${BENCH_FILES[@]}"; do
  if [[ ! -f "${bench}" ]]; then
    echo "benchmark 文件不存在: ${bench}"
    exit 1
  fi
done

if [[ -z "${API_BASE}" || -z "${API_KEY}" || -z "${API_MODEL}" ]]; then
  echo "API_BASE / API_KEY / API_MODEL 不能为空。"
  exit 1
fi

api_model_tag="$(sanitize_name "${API_MODEL}")"
OUTPUT_DIR="${OUTPUT_ROOT}/${api_model_tag}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Main Experiment (API)"
echo "API Base: ${API_BASE}"
echo "API Model: ${API_MODEL}"
echo "Output Root: ${OUTPUT_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

for bench in "${BENCH_FILES[@]}"; do
  bench_base="$(basename "${bench}" .jsonl)"
  output_file="${OUTPUT_DIR}/${bench_base}_api_eval.jsonl"

  echo "[RUN][main][api] benchmark=${bench}"
  echo "[RUN][main][api] output=${output_file}"

  python "${PY_SCRIPT}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --api_model "${API_MODEL}" \
    --benchmark_file "${bench}" \
    --output_file "${output_file}" \
    --max_tokens "${MAX_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --batch_size "${BATCH_SIZE}" \
    --max_prediction_retries "${MAX_PREDICTION_RETRIES}" \
    --image_url_mode "${IMAGE_URL_MODE}"
done

echo "[DONE] main experiment (api) 全部完成"
