# COHERENCE: Benchmarking Fine-Grained Image-Text Alignment in Interleaved Multimodal Contexts

This repository contains the evaluation code for **COHERENCE**.

## Links

- Dataset: https://huggingface.co/datasets/BingliW/COHERENCE
- ArXiv: (coming soon)
- Evaluation Code: this repository

## Introduction

In recent years, Multimodal Large Language Models (MLLMs) have achieved strong progress on many multimodal benchmarks. However, most existing benchmarks mainly evaluate single-image understanding, multi-image comparison, or general multimodal question answering. In real-world settings such as document reading, information is often presented as long interleaved image-text context. This requires models to not only understand each individual image, but also perform fine-grained image-text alignment and identify accurate correspondences between textual and visual content across long context.

In addition, models must integrate evidence across paragraphs and modalities for reasoning. Although this capability is important for practical applications, systematic benchmarks for quantifying fine-grained understanding in long interleaved image-text context are still limited.

To fill this gap, we propose **COHERENCE**, a benchmark designed to evaluate the ability of MLLMs to recover fine-grained image-text correspondences in long interleaved multimodal context. COHERENCE covers four representative domains and contains **7,670** high-quality questions. We also provide a six-type error analysis protocol for fine-grained attribution of failures in interleaved image-text understanding.

## Main Results

`Exact` means exact-match accuracy. `Partial` means Kendall-based partial score.

### Open-Source Models

| Model | WikiHow Exact | WikiHow Partial | StoryBird Exact | StoryBird Partial | Cooking Exact | Cooking Partial | Science Exact | Science Partial | Overall Exact | Overall Partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 VL 4B | 58.45 | 86.35 | 20.54 | 66.82 | 25.28 | 73.18 | 15.93 | 68.92 | 30.86 | 74.26 |
| Qwen3 VL 8B | 57.29 | 86.05 | 22.12 | 68.90 | 29.53 | 74.99 | 16.44 | 71.27 | 32.10 | 75.70 |
| Qwen3 VL 30B | 62.48 | 87.95 | 25.18 | 72.08 | 49.33 | 81.18 | 31.71 | 76.66 | 43.05 | 79.85 |
| Qwen3 VL 235B | 64.25 | 88.67 | 29.11 | 74.07 | 45.80 | 80.78 | 31.40 | 77.64 | 43.44 | 80.63 |
| Step3-VL 10B | 57.29 | 84.99 | 23.18 | 69.56 | 42.23 | 76.95 | 25.15 | 71.11 | 37.74 | 76.01 |
| GLM4.6V | 62.76 | 88.09 | 26.53 | 72.37 | 38.86 | 78.76 | 27.61 | 75.88 | 39.75 | 79.14 |
| Intern-S1-Pro | 64.06 | 88.77 | 26.47 | 72.31 | 52.12 | 81.58 | 33.81 | 77.08 | 45.01 | 80.33 |
| Kimi K2.5 | 75.43 | 93.23 | 41.84 | 81.14 | 57.31 | 82.92 | 50.15 | 84.00 | 56.98 | 85.60 |
| Qwen3.5 4B | 62.57 | 88.64 | 32.45 | 76.55 | 41.71 | 77.90 | 29.56 | 76.11 | 42.23 | 80.06 |
| Qwen3.5 35B-A3 | 69.53 | 90.71 | 42.96 | 81.00 | 53.01 | 81.73 | 41.29 | 81.66 | 52.28 | 83.99 |
| Qwen3.5 122B-A10 | 71.69 | 91.95 | 44.72 | 82.00 | 61.55 | 84.31 | 47.64 | 83.47 | 57.03 | 85.66 |
| Qwen3.5 397B-A17 | 69.63 | 91.16 | 49.53 | 83.84 | 69.79 | 87.44 | 57.99 | 86.51 | 62.24 | 87.41 |

### Closed-Source Models

| Model | WikiHow Exact | WikiHow Partial | StoryBird Exact | StoryBird Partial | Cooking Exact | Cooking Partial | Science Exact | Science Partial | Overall Exact | Overall Partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| doubao-seed-2-0-mini-260215 | 70.59 | 91.65 | 34.33 | 77.33 | 42.54 | 78.33 | 34.99 | 79.04 | 46.41 | 81.90 |
| doubao-seed-2-0-lite-260215 | 72.50 | 92.31 | 39.85 | 80.50 | 49.43 | 79.31 | 43.85 | 81.30 | 52.15 | 83.61 |
| doubao-seed-2-0-pro-260215 | 76.34 | 93.62 | 46.65 | 81.25 | 60.78 | 83.16 | 47.13 | 81.96 | 58.40 | 85.27 |
| Claude-sonnet-4-6-thinking | 73.90 | 92.67 | 47.71 | 83.35 | 69.90 | 87.26 | 56.51 | 85.95 | 62.65 | 87.53 |
| GPT-5.2-none | 68.28 | 90.05 | 33.63 | 75.75 | 53.21 | 83.14 | 39.45 | 80.24 | 49.45 | 82.64 |
| GPT-5.4-high | 77.26 | 93.78 | 51.59 | 84.68 | 77.41 | 90.47 | 66.65 | 89.44 | 67.25 | 89.05 |
| Gemini-3.1-pro-preview-thinking | 80.52 | 94.69 | 56.87 | 86.57 | 72.23 | 88.61 | 64.45 | 87.93 | 69.09 | 89.64 |

## Difficulty Results

`Exact` means exact-match accuracy. `Kendall` means Kendall-based partial score.

### Open-Source Models

| Model | Easy Exact | Easy Kendall | Medium Exact | Medium Kendall | Hard Exact | Hard Kendall | Overall Exact | Overall Kendall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 VL 4B | 42.08 | 75.49 | 18.86 | 74.40 | 2.82 | 64.20 | 30.86 | 74.26 |
| Qwen3 VL 8B | 42.33 | 76.02 | 21.49 | 76.12 | 4.93 | 71.14 | 32.10 | 75.70 |
| Qwen3 VL 30B | 54.52 | 80.94 | 31.86 | 79.31 | 9.15 | 74.03 | 43.05 | 79.85 |
| Qwen3 VL 235B | 54.32 | 81.19 | 33.10 | 80.58 | 10.04 | 76.56 | 43.44 | 80.63 |
| Step3-VL 10B | 50.30 | 77.80 | 25.09 | 75.31 | 2.64 | 65.61 | 37.74 | 76.01 |
| GLM4.6V | 50.60 | 79.34 | 28.95 | 79.44 | 8.80 | 76.21 | 39.75 | 79.14 |
| Intern-S1-Pro | 57.02 | 81.88 | 33.94 | 79.76 | 6.34 | 71.25 | 45.01 | 80.33 |
| Kimi K2.5 | 66.64 | 86.33 | 48.80 | 85.22 | 22.36 | 81.84 | 56.98 | 85.60 |
| Qwen3.5 4B | 52.85 | 80.81 | 31.90 | 79.83 | 10.74 | 75.42 | 42.23 | 80.06 |
| Qwen3.5 35B-A3 | 62.53 | 84.72 | 43.04 | 83.67 | 18.31 | 79.97 | 52.28 | 83.99 |
| Qwen3.5 122B-A10 | 67.77 | 86.76 | 48.03 | 85.01 | 18.13 | 80.38 | 57.03 | 85.66 |
| Qwen3.5 397B-A17 | 70.96 | 88.26 | 55.46 | 87.12 | 28.17 | 82.37 | 62.24 | 87.41 |

### Closed-Source Models

| Model | Easy Exact | Easy Kendall | Medium Exact | Medium Kendall | Hard Exact | Hard Kendall | Overall Exact | Overall Kendall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| doubao-seed-2-0-mini-260215 | 57.60 | 82.62 | 35.83 | 81.58 | 11.80 | 78.01 | 46.41 | 81.90 |
| doubao-seed-2-0-lite-260215 | 62.74 | 84.60 | 42.53 | 82.88 | 17.43 | 79.60 | 52.15 | 83.61 |
| doubao-seed-2-0-pro-260215 | 69.05 | 86.93 | 48.91 | 84.22 | 22.54 | 77.62 | 58.40 | 85.27 |
| Claude-sonnet-4-6-thinking | 73.30 | 88.88 | 53.75 | 86.47 | 23.94 | 82.27 | 62.65 | 87.53 |
| GPT-5.2-none | 61.27 | 84.18 | 38.24 | 81.49 | 13.03 | 76.33 | 49.45 | 82.64 |
| GPT-5.4-high | 75.06 | 90.82 | 54.06 | 86.74 | 24.45 | 80.87 | 62.88 | 88.44 |
| Gemini-3.1-pro-preview-thinking | 80.39 | 92.08 | 59.76 | 87.60 | 27.46 | 80.77 | 69.09 | 89.64 |

## Repository Structure

```text
.
├── main_experiment/
│   ├── evaluate_arrangement_vllm.py
│   ├── evaluate_arrangement_api.py
│   ├── run_main_vllm.sh
│   ├── run_main_api.sh
│   ├── error_analysis.py
│   ├── metrics.py
│   ├── stats_accuracy_by_domain.py
│   └── stats_accuracy_by_difficulty.py
└── ablation_experiment/
    ├── evaluate_arrangement_ablation_vllm.py
    ├── run_ablation_text_only.sh
    └── run_ablation_image_only.sh
```

## Installation

- Python 3.10+
- Linux with GPU for vLLM evaluation

```bash
pip install -U vllm transformers pillow tqdm openai
# Optional for some Qwen-VL setups
pip install -U qwen-vl-utils
```

## Required Configuration

Set image root paths in these files:

- `main_experiment/evaluate_arrangement_vllm.py`
- `main_experiment/evaluate_arrangement_api.py`
- `main_experiment/error_analysis.py`

Replace:

```python
IMAGES_ROOT = "Your Path Here"
```

with your real absolute image directory.

Also configure model paths:

- `main_experiment/run_main_vllm.sh` (`MODEL_PATH`)
- `ablation_experiment/run_ablation_text_only.sh` (`MODEL_PATH`)
- `ablation_experiment/run_ablation_image_only.sh` (`MODEL_PATH`)

## Benchmark Data Layout

Default scripts expect benchmark files under:

```text
datasets/benchmark_data/full_benchmark_7670/
```

including:

- `cooking_full.reasonable.jsonl`
- `science_full.reasonable.jsonl`
- `storybird_full.reasonable.jsonl`
- `wikihow_full.reasonable.jsonl`

## Evaluation

### vLLM: run all 4 subsets

```bash
bash main_experiment/run_main_vllm.sh main_experiment/results
```

### API: run all 4 subsets

```bash
export API_BASE="https://your-api-base/v1"
export API_KEY="your_api_key"
export API_MODEL="your_model_name"

bash main_experiment/run_main_api.sh main_experiment/results
```

### vLLM single-file example

```bash
python main_experiment/evaluate_arrangement_vllm.py \
  --model_path /path/to/your/model \
  --benchmark_file datasets/benchmark_data/full_benchmark_7670/cooking_full.reasonable.jsonl \
  --output_file main_experiment/results/your_model/cooking_full.reasonable_vllm_eval.jsonl \
  --max_tokens 12000 \
  --temperature 0.6 \
  --top_p 1.0 \
  --max_model_len 40000 \
  --batch_size 256 \
  --tensor_parallel_size 1 \
  --data_parallel_size 1
```

## Ablation

### Text-only

```bash
bash ablation_experiment/run_ablation_text_only.sh results/ablation_text_only
```

### Image-only

```bash
bash ablation_experiment/run_ablation_image_only.sh results/ablation_image_only
```

## Output Format

For each output jsonl, scripts also write:

- `<output>.summary.json`
- (vLLM main only) `<output>.dropped.jsonl`

Typical record fields:

- `dataset_type`, `data_id`, `url_id`, `title`
- `answer`, `prediction`
- `exact_correct`
- `partial_score` with metric `kendall_tau_0_1`
- `raw_input`, `model_input`, `raw_output`

## Error Analysis

```bash
python main_experiment/error_analysis.py --help
```

## Notes

- Evaluation supports resume mode if output files already exist.
- In current scripts, `data_parallel_size` is kept as a compatibility argument and may be forced to `1`.
- `stats_accuracy_by_domain.py` and `stats_accuracy_by_difficulty.py` import `accuracy_table_common`, which is not included in this snapshot.

## Citation

If you use COHERENCE, please cite:

**COHERENCE: Benchmarking Fine-Grained Image-Text Alignment in Interleaved Multimodal Contexts**

BibTeX will be added after paper release.

## License

This repository is licensed under **ODC Attribution License (ODC-By) 1.0**.
See [LICENSE](./LICENSE).
