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

In recent years, Multimodal Large Language Models (MLLMs) have achieved strong progress on many multimodal benchmarks. However, most existing benchmarks mainly evaluate single-image understanding, multi-image comparison, or general multimodal question answering. In real-world settings such as document reading, information is often presented as interleaved image-text context. This requires models to not only understand each individual image, but also perform fine-grained image-text alignment and identify accurate correspondences between textual and visual content across context.

In addition, models must integrate evidence across paragraphs and modalities for reasoning. Although this capability is important for practical applications, systematic benchmarks for quantifying fine-grained understanding in interleaved image-text context are still limited.

To fill this gap, we propose **COHERENCE**, a benchmark designed to evaluate the ability of MLLMs to recover fine-grained image-text correspondences in interleaved multimodal context. COHERENCE covers four representative domains and contains **7,670** high-quality questions. We also provide a six-type error analysis protocol for fine-grained attribution of failures in interleaved image-text understanding.

## Results Snapshot

To keep this README clean, we provide concise highlights instead of large tables.
(`Exact` = exact-match accuracy, `Partial/Kendall` = Kendall-based partial score.)

### Main Results

- Best open-source overall model: `Qwen3.5 397B-A17` with `64.81 Exact / 88.37 Partial`.
- Best closed-source overall model: `Gemini-3.1-pro-preview-thinking` with `71.82 Exact / 90.11 Partial`.
- Strong closed-source runner-up: `GPT-5.4-high` with `71.29 Exact / 86.54 Partial`.
- Domain bests (closed-source): `Gemini-3.1-pro-preview-thinking` leads on `WikiHow` and `StoryBird`, while `GPT-5.4-high` leads on `Cooking` and `Science`.

## Repository Structure

```text
.
├── code/
│   ├── main_experiment/
│   │   ├── evaluate_arrangement_vllm.py
│   │   ├── evaluate_arrangement_api.py
│   │   ├── run_main_vllm.sh
│   │   ├── run_main_api.sh
│   │   ├── error_analysis.py
│   │   ├── metrics.py
│   │   ├── stats_accuracy_by_domain.py
│   │   └── stats_accuracy_by_difficulty.py
│   └── ablation_experiment/
│       ├── evaluate_arrangement_ablation_vllm.py
│       ├── run_ablation_text_only.sh
│       └── run_ablation_image_only.sh
└── datasets/  # created after download
```

## Installation

```bash
pip install -U vllm transformers pillow tqdm openai
# Optional for some Qwen-VL setups
pip install -U qwen-vl-utils
```

## Step-by-Step Run Guide

Use one of the two routes below:

- Route A: API evaluation (`openai` SDK)
- Route B: local vLLM evaluation (`vllm`)

### Step 1. Create environment

For API route:

```bash
python -m venv .venv-api
source .venv-api/bin/activate
pip install -U openai transformers pillow tqdm
```

For vLLM route:

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
pip install -U vllm transformers pillow tqdm
# Optional for some Qwen-VL setups
pip install -U qwen-vl-utils
```

### Step 2. Download dataset

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download BingliW/COHERENCE \
  --repo-type dataset \
  --local-dir datasets
```

Current defaults are already aligned with the above download command.

### Step 3. Minimal config

- API route: set `API_BASE`, `API_KEY`, `API_MODEL`.
- vLLM route / ablation: set `MODEL_PATH` in the corresponding `.sh` script.
- Optional only if you use a custom data layout: adjust `JSONL_DIR` and/or `IMAGES_ROOT`.

### Step 4A. Run API evaluation

```bash
export API_BASE="https://your-api-base/v1"
export API_KEY="your_api_key"
export API_MODEL="your_model_name"

bash code/main_experiment/run_main_api.sh code/main_experiment/results
```

### Step 4B. Run vLLM evaluation

```bash
# Optional: export TP_SIZE=1 (or another value)
bash code/main_experiment/run_main_vllm.sh code/main_experiment/results
```

After running, predictions are written to `code/main_experiment/results/...` and each jsonl has a matching `.summary.json`.

## Ablation

### Text-only

```bash
bash code/ablation_experiment/run_ablation_text_only.sh code/results/ablation_text_only
```

### Image-only

```bash
bash code/ablation_experiment/run_ablation_image_only.sh code/results/ablation_image_only
```

## Output Format

For each output jsonl, scripts also write:

- `<output>.summary.json`

Typical record fields:

- `dataset_type`, `data_id`, `url_id`, `title`
- `answer`, `prediction`
- `exact_correct`
- `partial_score` with metric `kendall_tau_0_1`
- `raw_input`, `model_input`, `raw_output`

## Error Analysis

```bash
python code/main_experiment/error_analysis.py --help
```

## Notes

- Evaluation supports resume mode if output files already exist.
- Current vLLM scripts expose `tensor_parallel_size` only.
- `stats_accuracy_by_domain.py` and `stats_accuracy_by_difficulty.py` import `accuracy_table_common`, which is not included in this snapshot.

## Citation

If you use COHERENCE, please cite:

**COHERENCE: Benchmarking Fine-Grained Image-Text Alignment in Interleaved Multimodal Contexts**

BibTeX will be added after paper release.

## License

This repository is licensed under **ODC Attribution License (ODC-By) 1.0**.
See [LICENSE](./LICENSE).
