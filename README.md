# COHERENCE: Benchmarking Fine-Grained Image-Text Alignment in Interleaved Multimodal Contexts

This repository contains the evaluation code for **COHERENCE**.

<p align="center">
  <a href="https://huggingface.co/datasets/BingliW/COHERENCE">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-COHERENCE-fcc624?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Dataset">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/%F0%9F%93%9A%20Paper-Coming%20Soon-4c78ff?style=for-the-badge" alt="Paper Coming Soon">
  </a>
  <a href=".">
    <img src="https://img.shields.io/badge/%F0%9F%90%99%20GitHub-Evaluation%20Code-181717?style=for-the-badge&logo=github" alt="GitHub Evaluation Code">
  </a>
</p>

## Introduction

In recent years, Multimodal Large Language Models (MLLMs) have achieved strong progress on many multimodal benchmarks. However, most existing benchmarks mainly evaluate single-image understanding, multi-image comparison, or general multimodal question answering. In real-world settings such as document reading, information is often presented as long interleaved image-text context. This requires models to not only understand each individual image, but also perform fine-grained image-text alignment and identify accurate correspondences between textual and visual content across long context.

In addition, models must integrate evidence across paragraphs and modalities for reasoning. Although this capability is important for practical applications, systematic benchmarks for quantifying fine-grained understanding in long interleaved image-text context are still limited.

To fill this gap, we propose **COHERENCE**, a benchmark designed to evaluate the ability of MLLMs to recover fine-grained image-text correspondences in long interleaved multimodal context. COHERENCE covers four representative domains and contains **7,670** high-quality questions. We also provide a six-type error analysis protocol for fine-grained attribution of failures in interleaved image-text understanding.

## Results Snapshot

To keep this README clean, we provide concise highlights instead of large tables.
(`Exact` = exact-match accuracy, `Partial/Kendall` = Kendall-based partial score.)

### Main Results (Domain-Level)

- Best open-source overall model: `Qwen3.5 397B-A17` with `62.24 Exact / 87.41 Partial`.
- Best closed-source overall model: `Gemini-3.1-pro-preview-thinking` with `69.09 Exact / 89.64 Partial`.
- Strong closed-source runner-up: `GPT-5.4-high` with `67.25 Exact / 89.05 Partial`.
- Domain bests (closed-source): `Gemini-3.1-pro-preview-thinking` leads on `WikiHow` and `StoryBird`, while `GPT-5.4-high` leads on `Cooking` and `Science`.

### Difficulty Results

- Best open-source overall by difficulty: `Qwen3.5 397B-A17` with `62.24 Exact / 87.41 Kendall`.
- Best closed-source overall by difficulty: `Gemini-3.1-pro-preview-thinking` with `69.09 Exact / 89.64 Kendall`.
- Hard split remains challenging across all families; best hard exact scores are around `27-28%`.
- COHERENCE still shows a clear gap between Easy/Medium and Hard settings, indicating room for progress in long-context fine-grained alignment.

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
