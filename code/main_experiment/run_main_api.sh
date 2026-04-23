#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/evaluate_arrangement_api.py"
JSONL_DIR="../../datasets/jsonl"
OUTPUT_ROOT="${1:-${SCRIPT_DIR}/results}"

API_BASE="${API_BASE:-<API_BASE>}"
API_KEY="${API_KEY:-<API_KEY>}"
API_MODEL="${API_MODEL:-<API_MODEL>}"

BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_TOKENS="${MAX_TOKENS:-6000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_PREDICTION_RETRIES="${MAX_PREDICTION_RETRIES:-10}"
IMAGE_URL_MODE="${IMAGE_URL_MODE:-auto}"

BENCH_FILES=(
  #"${JSONL_DIR}/cooking.jsonl"
  #"${JSONL_DIR}/science.jsonl"
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

if [[ "${API_BASE}" == "<API_BASE>" || "${API_KEY}" == "<API_KEY>" || "${API_MODEL}" == "<API_MODEL>" ]]; then
  echo "Please set API_BASE / API_KEY / API_MODEL before running."
  exit 1
fi

if [[ -z "${API_BASE}" || -z "${API_KEY}" || -z "${API_MODEL}" ]]; then
  echo "API_BASE / API_KEY / API_MODEL must not be empty."
  exit 1
fi

api_model_tag="$(sanitize_name "${API_MODEL}")"
OUTPUT_DIR="${OUTPUT_ROOT}/${api_model_tag}"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Main Experiment (API)"
echo "API Config: [REDACTED]"
echo "=========================================="

for bench in "${BENCH_FILES[@]}"; do
  bench_base="$(basename "${bench}" .jsonl)"
  output_file="${OUTPUT_DIR}/${bench_base}_api_eval.jsonl"

  echo "[RUN][main][api] benchmark=${bench_base}"

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

echo "[DONE] main experiment (api) all done"
