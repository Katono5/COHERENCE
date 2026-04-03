#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-modality ablation ranking evaluation (vLLM backend).

Supported ablation modes:
- text_only: keep article text and [IMAGE_PLACEHOLDER] only; do not provide candidate images.
- image_only: keep candidate images only; do not provide article body text.

"""

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

COHERENCE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_EXPERIMENT_DIR = os.path.join(COHERENCE_ROOT, "main_experiment")
if MAIN_EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, MAIN_EXPERIMENT_DIR)

import evaluate_arrangement_vllm as base_eval



def build_prompt_text_only(item: Dict[str, Any]) -> str:
    """Text-only ablation: do not provide images while avoiding "cannot see images" responses."""
    content = item["content"]
    num_placeholders = item["num_placeholders"]
    num_images = len(item["image_sequence"])

    return f"""## Task: Text-Only Ablation for Interleaved-Image-Text Matching

This is an intentional text-only evaluation setting.
- Candidate image contents are hidden by design.
- You must NOT say that images are missing or unavailable.
- Infer the best possible one-to-one mapping from text context alone.
- When uncertain, still output a complete answer list.

You are given an article with {num_placeholders} placeholders marked as [IMAGE_PLACEHOLDER].
There are {num_images} candidate image indices (Image 0, Image 1, ..., Image {num_images - 1}),
but their visual contents are intentionally not shown in this ablation.

## Article Text (with placeholders):

{content}

## Instructions:

1. Read the text around each [IMAGE_PLACEHOLDER].
2. Infer which candidate index is most likely for each placeholder.
3. Use each index at most once.

## Output Format:

First reason briefly, then output your final answer on the LAST line as a Python list:
- Format: [{", ".join(["index" + str(i) for i in range(num_placeholders)])}]
- The list position corresponds to placeholder order (first placeholder is index 0).
- Do NOT output inverse mapping (image -> placeholder).
- The list must have exactly {num_placeholders} integers, each between 0 and {num_images - 1}.

Now provide your final answer."""


def build_prompt_image_only_part1(item: Dict[str, Any]) -> str:
    """Image-only ablation: provide only images and request ranking without article text."""
    num_placeholders = item["num_placeholders"]
    num_images = len(item["image_sequence"])

    return f"""## Task: Image-Only Ablation for Interleaved-Image-Text Matching

This is an intentional image-only evaluation setting.
- The article text is hidden by design.
- You must infer the most plausible placeholder assignment using visual cues only.
- Prefer coherent temporal/procedural/narrative ordering across images.

There are {num_placeholders} placeholder positions to fill and {num_images} candidate images
(Image 0, Image 1, ..., Image {num_images - 1}) shown below.

## Candidate Images (Image 0 to Image {num_images - 1}):
"""


def build_prompt_image_only_part2(item: Dict[str, Any]) -> str:
    num_placeholders = item["num_placeholders"]
    num_images = len(item["image_sequence"])

    return f"""

## Instructions:

1. Analyze each candidate image carefully.
2. Infer a best-guess mapping from placeholder order to image indices.
3. The same index can only be used once.

## Output Format:

First reason briefly, then output your final answer on the LAST line as a Python list:
- Format: [{", ".join(["index" + str(i) for i in range(num_placeholders)])}]
- The list position corresponds to placeholder order (first placeholder is index 0).
- Do NOT output inverse mapping (image -> placeholder).
- The list must have exactly {num_placeholders} integers, each between 0 and {num_images - 1}.

Now provide your final answer."""


def build_messages_text_only(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    normalized = base_eval.normalize_item_for_vllm(item)
    prompt = build_prompt_text_only(normalized)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    raw_input = {
        "prompt": prompt,
        "images": [],
    }
    return messages, raw_input, normalized


def build_messages_image_only(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    normalized = base_eval.normalize_item_for_vllm(item)
    image_paths = base_eval.resolve_image_paths(normalized["image_sequence"])

    part1 = build_prompt_image_only_part1(normalized)
    part2 = build_prompt_image_only_part2(normalized)

    content: List[Dict[str, Any]] = [{"type": "text", "text": part1}]
    for idx, full_path in enumerate(image_paths):
        content.append({"type": "text", "text": f"Image {idx}:"})
        content.append({"type": "image", "image": full_path, "url": full_path})
    content.append({"type": "text", "text": part2})

    prompt_text = "\n".join([part1] + [f"Image {i}:" for i in range(len(image_paths))] + [part2])
    raw_input = {
        "prompt": prompt_text,
        "images": image_paths,
    }
    return [{"role": "user", "content": content}], raw_input, normalized


def build_messages_for_ablation(
    item: Dict[str, Any],
    ablation_mode: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    if ablation_mode == "text_only":
        return build_messages_text_only(item)
    if ablation_mode == "image_only":
        return build_messages_image_only(item)
    raise ValueError(f"unknown ablation_mode: {ablation_mode}")


def _extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append(str(part.get("text", "")))
    return "\n".join(parts)


def prepare_inputs_for_ablation(
    messages: List[Dict[str, Any]],
    processor: Any,
    model_path: str,
    ablation_mode: str,
) -> Dict[str, Any]:
    if ablation_mode == "text_only":
        try:
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return {"prompt": prompt}
        except Exception:
            return {"prompt": _extract_text_from_messages(messages)}

    return base_eval.prepare_inputs_for_vllm(messages, processor, model_path)


def evaluate_ablation_vllm(
    model_path: str,
    benchmark_items: List[Dict[str, Any]],
    output_file: str,
    ablation_mode: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_model_len: int = 16384,
    batch_size: int = 8,
    tensor_parallel_size: int = 8,
    disable_mm_preprocessor_cache: bool = True,
    max_prediction_retries: int = 10,
) -> Dict[str, Any]:
    from vllm import LLM, SamplingParams

    if ablation_mode not in {"text_only", "image_only"}:
        raise ValueError("ablation_mode must be one of: text_only, image_only")
    if max_prediction_retries < 0:
        raise ValueError("max_prediction_retries must be >= 0")

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    base_llm_kwargs: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "tensor_parallel_size": tensor_parallel_size,
    }
    if disable_mm_preprocessor_cache:
        base_llm_kwargs["mm_processor_cache_gb"] = 0
        base_llm_kwargs["disable_mm_preprocessor_cache"] = True

    processor = base_eval.load_processor_with_compat(model_path)
    base_eval.validate_glm_runtime_compat(model_path, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    base_eval.ensure_results_dir(output_file)

    total = len(benchmark_items)
    dropped_total = 0
    exact_correct = 0
    partial_score_sum = 0.0
    evaluated_total = 0

    max_prediction_attempts = max_prediction_retries + 1

    with open(output_file, "w", encoding="utf-8") as out_f:
        print(f"[{ablation_mode}] [Stage-2] Inference started, total samples: {total}")
        llm, resolved_llm_kwargs = base_eval.build_llm_with_kwarg_compat(LLM, base_llm_kwargs)

        num_batches = (len(benchmark_items) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(benchmark_items))
            batch_items = benchmark_items[start:end]

            batch_samples: List[Dict[str, Any]] = []
            for item in batch_items:
                try:
                    messages, raw_input, normalized = build_messages_for_ablation(item, ablation_mode)
                    sample_input = prepare_inputs_for_ablation(
                        messages=messages,
                        processor=processor,
                        model_path=model_path,
                        ablation_mode=ablation_mode,
                    )
                    batch_samples.append(
                        {
                            "item": item,
                            "raw_input": raw_input,
                            "normalized": normalized,
                            "sample_input": sample_input,
                        }
                    )
                except Exception as build_err:
                    dropped_total += 1
                    reason = (
                        "bad_image_inference"
                        if base_eval.is_bad_image_error(build_err)
                        else "input_prepare_exception"
                    )
                    print(
                        f"[{ablation_mode}] batch {batch_idx + 1}/{num_batches} input build failed; skipped 1 sample: "
                        f"reason={reason}, data_id={item.get('data_id')}, "
                        f"error={type(build_err).__name__}: {build_err}"
                    )

            if not batch_samples:
                print(
                    f"[{ablation_mode}] [Stage-2] Completed batch {batch_idx + 1}/{num_batches}, "
                    f"no inferable samples in this batch (total skipped: {dropped_total})"
                )
                continue

            batch_inputs = [sample["sample_input"] for sample in batch_samples]
            try:
                outputs = list(llm.generate(batch_inputs, sampling_params=sampling_params))
            except Exception as err:
                if base_eval.is_mm_cache_assertion(err):
                    dropped_total += len(batch_samples)
                    print(
                        f"[{ablation_mode}] batch {batch_idx + 1}/{num_batches} hit mm_hash assertion, "
                        f"skipping this batch ({len(batch_samples)} samples) and rebuilding engine."
                    )
                    try:
                        del llm
                        llm, _ = base_eval.build_llm_with_kwarg_compat(LLM, resolved_llm_kwargs)
                    except Exception:
                        raise
                    continue
                raise

            sample_states: List[Dict[str, Any]] = []
            for sample_idx_in_batch, sample in enumerate(batch_samples):
                generated_text = ""
                if sample_idx_in_batch < len(outputs):
                    output = outputs[sample_idx_in_batch]
                    if output is not None and getattr(output, "outputs", None):
                        generated_text = output.outputs[0].text or ""

                sample_states.append(
                    {
                        "item": sample["item"],
                        "raw_input": sample["raw_input"],
                        "normalized": sample["normalized"],
                        "sample_input": sample["sample_input"],
                        "generated_text": generated_text,
                        "attempts_used": 1,
                        "retry_errors": [],
                    }
                )

            pending_indices = [
                idx
                for idx, state in enumerate(sample_states)
                if base_eval.parse_prediction_list(state["generated_text"]) is None
            ]

            if pending_indices:
                print(
                    f"[{ablation_mode}] batch {batch_idx + 1}/{num_batches} after first pass: "
                    f"prediction=null {len(pending_indices)}/{len(sample_states)}, "
                    f"max retries: {max_prediction_retries}"
                )

            for attempt_idx in range(2, max_prediction_attempts + 1):
                if not pending_indices:
                    break

                retry_inputs = [sample_states[idx]["sample_input"] for idx in pending_indices]
                for idx in pending_indices:
                    sample_states[idx]["attempts_used"] = attempt_idx

                try:
                    retry_outputs = list(llm.generate(retry_inputs, sampling_params=sampling_params))
                except Exception as retry_err:
                    err_msg = f"attempt_{attempt_idx}: {type(retry_err).__name__}: {retry_err}"
                    if base_eval.is_mm_cache_assertion(retry_err):
                        err_msg = f"attempt_{attempt_idx}: mm_cache_assertion: {retry_err}"
                        try:
                            del llm
                            llm, _ = base_eval.build_llm_with_kwarg_compat(LLM, resolved_llm_kwargs)
                        except Exception:
                            raise

                    for idx in pending_indices:
                        sample_states[idx]["retry_errors"].append(err_msg)
                    continue

                next_pending_indices: List[int] = []
                for output_idx, sample_idx_in_states in enumerate(pending_indices):
                    generated_text = ""
                    if output_idx < len(retry_outputs):
                        retry_output = retry_outputs[output_idx]
                        if retry_output is not None and getattr(retry_output, "outputs", None):
                            generated_text = retry_output.outputs[0].text or ""

                    sample_states[sample_idx_in_states]["generated_text"] = generated_text
                    if base_eval.parse_prediction_list(generated_text) is None:
                        next_pending_indices.append(sample_idx_in_states)

                pending_indices = next_pending_indices

            # Record batch-level results.
            for state in sample_states:
                item = state["item"]
                answer = item["answer"]
                normalized = state["normalized"]
                pred_list = base_eval.parse_prediction_list(state["generated_text"])

                if pred_list is None:
                    exact = 0.0
                    partial = 0.0
                else:
                    exact = 1.0 if base_eval.exact_match(pred_list, answer) else 0.0
                    partial = base_eval.partial_match(
                        pred_list,
                        answer,
                        normalized["num_placeholders"],
                    )

                exact_correct += int(exact)
                partial_score_sum += partial
                evaluated_total += 1

                model_input: Dict[str, Any] = {
                    "prediction_retry_max": max_prediction_retries,
                    "prediction_attempts": int(state["attempts_used"]),
                    "ablation_mode": ablation_mode,
                }
                if state["retry_errors"]:
                    model_input["retry_errors"] = list(state["retry_errors"])
                if pred_list is None:
                    model_input["error"] = "prediction_parse_failed"

                record = {
                    "dataset_type": item.get("dataset_type"),
                    "data_id": item.get("data_id"),
                    "url_id": item.get("url_id"),
                    "title": item.get("title"),
                    "ablation_mode": ablation_mode,
                    "answer": answer,
                    "prediction": pred_list,
                    "raw_input": state["raw_input"],
                    "model_input": model_input,
                    "raw_output": state["generated_text"],
                    "exact_correct": bool(exact == 1.0),
                    "partial_score": partial,
                    "partial_metric": "kendall_tau_0_1",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            base_eval.flush_and_sync(out_f)
            print(
                f"[{ablation_mode}] [Stage-2] Completed batch {batch_idx + 1}/{num_batches}, "
                f"processed {end}/{len(benchmark_items)} in total, skipped {dropped_total} in total"
            )

    if evaluated_total <= 0:
        raise ValueError(f"[{ablation_mode}] No valid sample left after inference")

    exact_acc = exact_correct / evaluated_total
    partial_acc = partial_score_sum / evaluated_total

    print(
        f"[{ablation_mode}] Evaluation finished: original samples={total}, valid samples={evaluated_total}, "
        f"removed={dropped_total}, exact_acc={exact_acc:.4f}, partial_acc={partial_acc:.4f}"
    )

    return {
        "mode": "vllm_ablation",
        "ablation_mode": ablation_mode,
        "model_path": model_path,
        "total": evaluated_total,
        "original_total": total,
        "exact_correct": int(exact_correct),
        "exact_accuracy": exact_acc,
        "partial_accuracy": partial_acc,
        "dropped_samples": dropped_total,
        "output_file": output_file,
        "tensor_parallel_size": tensor_parallel_size,
    }


def _output_with_mode_suffix(output_file: str, mode: str) -> str:
    if output_file.endswith(".jsonl"):
        return output_file[: -len(".jsonl")] + f".{mode}.jsonl"
    return output_file + f".{mode}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Interleaved-Image-Text single-modality ablation vLLM evaluation script")
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to benchmark jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output result jsonl")
    parser.add_argument(
        "--ablation_mode",
        type=str,
        choices=["text_only", "image_only", "both"],
        default="both",
        help="Single-modality ablation mode",
    )
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--max_prediction_retries",
        type=int,
        default=10,
        help="Maximum retry rounds when prediction parses as None (excluding the first attempt)",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument(
        "--disable_mm_preprocessor_cache",
        action="store_true",
        default=True,
        help="Enabled by default: disable vLLM multimodal preprocessor cache",
    )
    parser.add_argument(
        "--enable_mm_preprocessor_cache",
        dest="disable_mm_preprocessor_cache",
        action="store_false",
        help="Explicitly enable vLLM multimodal preprocessor cache",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    items = base_eval.load_benchmark(args.benchmark_file)

    run_modes = [args.ablation_mode] if args.ablation_mode != "both" else ["text_only", "image_only"]
    all_summaries: Dict[str, Any] = {}

    for mode in run_modes:
        mode_output_file = args.output_file
        if args.ablation_mode == "both":
            mode_output_file = _output_with_mode_suffix(args.output_file, mode)

        summary = evaluate_ablation_vllm(
            model_path=args.model_path,
            benchmark_items=items,
            output_file=mode_output_file,
            ablation_mode=mode,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
            max_prediction_retries=args.max_prediction_retries,
        )

        summary_path = mode_output_file + ".summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[{mode}] Summary saved to: {summary_path}")

        all_summaries[mode] = summary

    if args.ablation_mode == "both":
        merged_summary_path = args.output_file + ".ablation.summary.json"
        with open(merged_summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"[both] Merged summary saved to: {merged_summary_path}")


if __name__ == "__main__":
    main()
