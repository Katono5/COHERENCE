#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COHERENCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PY_SCRIPT="${SCRIPT_DIR}/evaluate_arrangement_ablation_vllm.py"
MODEL_PATH="/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/zskj-hub/models--Qwen--Qwen3.5-397B-A17B/"
BENCH_DIR="${COHERENCE_ROOT}/datasets/benchmark_data/full_benchmark_7670"
OUTPUT_DIR="${1:-${COHERENCE_ROOT}/results/ablation_image_only/Qwen3.5}"

TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40000}"
MAX_TOKENS="${MAX_TOKENS:-12120}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-1.0}"
MAX_PREDICTION_RETRIES="${MAX_PREDICTION_RETRIES:-0}"

BENCH_FILES=(
  "${BENCH_DIR}/cooking_full.reasonable.jsonl"
  "${BENCH_DIR}/science_full.reasonable.jsonl"
  "${BENCH_DIR}/storybird_full.reasonable.jsonl"
  "${BENCH_DIR}/wikihow_full.reasonable.jsonl"
)

mkdir -p "${OUTPUT_DIR}"

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

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在: ${MODEL_PATH}"
  echo "请把 Qwen3.5-397B 放到 COHERENCE/models/Qwen3.5-397B，或通过 MODEL_PATH 环境变量指定。"
  exit 1
fi

echo "=========================================="
echo "Ablation Experiment: image_only"
echo "Model: ${MODEL_PATH}"
echo "TP=${TP_SIZE}, DP=${DP_SIZE}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=========================================="

for bench in "${BENCH_FILES[@]}"; do
  bench_base="$(basename "${bench}" .jsonl)"
  output_file="${OUTPUT_DIR}/Qwen3.5-397B_${bench_base}_ablation_image_only.jsonl"

  echo "[RUN][image_only] benchmark=${bench}"
  echo "[RUN][image_only] output=${output_file}"

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
    --tensor_parallel_size "${TP_SIZE}" \
    --data_parallel_size "${DP_SIZE}"
done

echo "[DONE] image_only ablation 全部完成"
