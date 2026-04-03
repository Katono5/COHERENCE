#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from transformers import AutoProcessor, __version__ as TRANSFORMERS_VERSION

try:
    from metrics import kendall_tau_mapped_0_1
except ImportError:
    from main_experiment.metrics import kendall_tau_mapped_0_1

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

# Image root directory
IMAGES_ROOT = "IMAGE PATH HERE"
PLACEHOLDER = "[IMAGE_PLACEHOLDER]"


def is_mm_cache_assertion(err: BaseException) -> bool:
    return "Expected a cached item for mm_hash=" in str(err)


def make_item_key(item: Dict[str, Any]) -> str:
    """Prefer deduplication by data_id; fall back to a composite key when missing."""
    data_id = str(item.get("data_id", "")).strip()
    if data_id:
        return f"data_id:{data_id}"
    return "fallback:" + "|".join(
        [
            str(item.get("dataset_type", "")),
            str(item.get("url_id", "")),
            str(item.get("title", "")),
        ]
    )


def _parse_exact_correct(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:  # nan
            return None
        return float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        if parsed != parsed:  # nan
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _partial_from_lists(pred: Any, answer: Any) -> float:
    return kendall_tau_mapped_0_1(pred, answer)


def load_existing_eval_results(output_file: str) -> Dict[str, Any]:
    if not os.path.exists(output_file):
        return {"count": 0, "exact_sum": 0.0, "partial_sum": 0.0, "keys": set(), "null_count": 0}

    latest_records: Dict[str, Dict[str, Any]] = {}
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = make_item_key(record)
            # When the same key appears multiple times, keep the last record as final.
            latest_records[key] = record

    count = len(latest_records)
    exact_sum = 0.0
    partial_sum = 0.0
    null_count = 0
    keys: Set[str] = set(latest_records.keys())

    for record in latest_records.values():
        pred = record.get("prediction")
        answer = record.get("answer")

        if pred is None:
            null_count += 1

        parsed_exact = None
        if "exact_correct" in record:
            parsed_exact = _parse_exact_correct(record.get("exact_correct"))
        if parsed_exact is None and isinstance(pred, list) and isinstance(answer, list):
            parsed_exact = pred == answer
        exact_sum += 1.0 if parsed_exact else 0.0

        partial_metric = str(record.get("partial_metric", "")).strip().lower()
        partial_score = (
            _to_float_or_none(record.get("partial_score"))
            if partial_metric == "kendall_tau_0_1"
            else None
        )
        if partial_score is None:
            partial_score = _partial_from_lists(pred, answer)
        partial_sum += partial_score

    return {
        "count": count,
        "exact_sum": exact_sum,
        "partial_sum": partial_sum,
        "keys": keys,
        "null_count": null_count,
    }


def prepare_output_file_for_resume(output_file: str) -> Dict[str, int]:
    """Compact historical results and keep only the latest record per key; requeue latest=null cases."""
    if not os.path.exists(output_file):
        return {"requeued_count": 0, "kept_count": 0}

    parsed_lines: List[Tuple[str, Optional[str], Optional[Dict[str, Any]]]] = []
    latest_records: Dict[str, Dict[str, Any]] = {}
    latest_line_idx: Dict[str, int] = {}
    valid_line_count = 0

    with open(output_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parsed_lines.append((raw_line, None, None))
                continue

            key = make_item_key(record)
            parsed_lines.append((raw_line, key, record))
            latest_records[key] = record
            latest_line_idx[key] = len(parsed_lines) - 1
            valid_line_count += 1

    requeued_keys = {
        key for key, record in latest_records.items() if record.get("prediction") is None
    }
    duplicate_count = max(0, valid_line_count - len(latest_records))
    if not requeued_keys and duplicate_count == 0:
        return {"requeued_count": 0, "kept_count": len(latest_records)}

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (raw_line, key, _record) in enumerate(parsed_lines):
            if key is None:
                f.write(raw_line)
                continue
            if idx != latest_line_idx.get(key):
                continue
            if key in requeued_keys:
                continue
            f.write(raw_line)

    return {
        "requeued_count": len(requeued_keys),
        "kept_count": len(latest_records) - len(requeued_keys),
    }


def collect_pending_items(items: List[Dict[str, Any]], skip_keys: Set[str]) -> Tuple[List[Dict[str, Any]], int]:
    pending_items: List[Dict[str, Any]] = []
    seen_pending_keys: Set[str] = set()
    duplicate_count = 0
    for item in items:
        key = make_item_key(item)
        if key in skip_keys:
            continue
        if key in seen_pending_keys:
            duplicate_count += 1
            continue
        seen_pending_keys.add(key)
        pending_items.append(item)
    return pending_items, duplicate_count


def is_bad_image_error(err: BaseException) -> bool:
    if isinstance(err, FileNotFoundError):
        return True

    if not isinstance(err, OSError):
        return False

    msg = str(err).lower()
    image_error_patterns = [
        "image file is truncated",
        "cannot identify image file",
        "broken data stream when reading image file",
        "no such file or directory",
        "truncated",
    ]
    return any(pattern in msg for pattern in image_error_patterns)


def _is_qwen_model(model_path: str) -> bool:
    return "qwen" in model_path.lower()


def _is_glm_model(model_path: str) -> bool:
    lower = model_path.lower()
    return "glm" in lower or "chatglm" in lower


def _major_version(version_str: str) -> Optional[int]:
    match = re.match(r"^\s*(\d+)", version_str)
    if match is None:
        return None
    return int(match.group(1))


def _collect_pil_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "image":
                continue
            image_path = part.get("image") or part.get("url")
            if not isinstance(image_path, str) or not image_path:
                continue
            with Image.open(image_path) as img:
                pil_images.append(img.convert("RGB").copy())
    return pil_images


def _resolve_patch_size(processor: Any) -> int:
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "patch_size"):
        patch_size = processor.image_processor.patch_size
    elif hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "image_processor_config"
    ):
        config = processor.image_processor.image_processor_config
        patch_size = getattr(config, "patch_size", None)
    else:
        patch_size = None

    if patch_size is None:
        raise AttributeError("processor.image_processor.patch_size is missing")
    return int(patch_size)


def _prepare_inputs_with_qwen_vision_utils(
    messages: List[Dict[str, Any]], processor: Any
) -> Dict[str, Any]:
    if process_vision_info is None:
        raise RuntimeError("qwen_vl_utils is not available")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    patch_size = _resolve_patch_size(processor)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if image_inputs is None:
        raise ValueError("image_inputs is None")
    mm_data: Dict[str, Any] = {"image": image_inputs}
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    prepared = {
        "prompt": text,
        "multi_modal_data": mm_data,
    }
    if video_kwargs is not None:
        prepared["mm_processor_kwargs"] = video_kwargs
    return prepared


def _prepare_inputs_generic(messages: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = _collect_pil_images_from_messages(messages)
    if not image_inputs:
        raise ValueError("no image found in messages")
    return {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs},
    }


def load_processor_with_compat(model_path: str) -> Any:
    processor_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
    }
    try:
        return AutoProcessor.from_pretrained(
            model_path,
            fix_mistral_regex=True,
            **processor_kwargs,
        )
    except TypeError:
        return AutoProcessor.from_pretrained(model_path, **processor_kwargs)


def validate_glm_runtime_compat(model_path: str, processor: Any) -> None:
    if not _is_glm_model(model_path):
        return

    major = _major_version(TRANSFORMERS_VERSION)
    has_image_processor = hasattr(processor, "image_processor")
    if major is not None and major < 5 and not has_image_processor:
        raise RuntimeError(
            "Detected a GLM model, but the current transformers version does not satisfy multimodal requirements: "
            f"current={TRANSFORMERS_VERSION}. Please upgrade to transformers>=5.0.0rc0 "
            "before running evaluate_arrangement_vllm.py."
        )


def build_llm_with_kwarg_compat(llm_cls: Any, llm_kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Handle vLLM kwarg compatibility by removing unsupported kwargs and retrying."""
    resolved_kwargs = dict(llm_kwargs)
    while True:
        try:
            llm = llm_cls(**resolved_kwargs)
            return llm, resolved_kwargs
        except TypeError as err:
            match = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", str(err))
            if not match:
                raise
            kwarg_name = match.group(1)
            if kwarg_name not in resolved_kwargs:
                raise
            print(f"Current vLLM version does not support `{kwarg_name}`; removed automatically and retrying.")
            resolved_kwargs.pop(kwarg_name, None)


def _extract_from_content_list(content_list: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Build text and image sequences from an interleaved content list."""
    content_parts: List[str] = []
    image_sequence: List[Dict[str, Any]] = []

    for part in content_list:
        part_type = part["type"]
        if part_type == "text":
            text = part["content"].strip()
            content_parts.append(text)
        elif part_type == "image":
            content_parts.append(PLACEHOLDER)
            image_sequence.append(
                {
                    "id": part["id"],
                    "image_path": part["image_path"],
                }
            )
        else:
            raise ValueError(f"Unknown content type: {part_type}")

    return "\n".join(content_parts), image_sequence


def normalize_item_for_vllm(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize item formats to content(str) + image_sequence(list)."""
    content = item["content"]
    if not isinstance(content, list):
        raise TypeError("content must be a list")
    content_text, image_sequence = _extract_from_content_list(content)
    num_placeholders = item["num_placeholders"]
    normalized = dict(item)
    normalized["content"] = content_text
    normalized["image_sequence"] = image_sequence
    normalized["num_placeholders"] = num_placeholders
    return normalized


def resolve_image_paths(image_sequence: List[Dict[str, Any]]) -> List[str]:
    image_paths: List[str] = []
    for img in image_sequence:
        rel_path = img.get("image_path", "")
        if not isinstance(rel_path, str):
            rel_path = str(rel_path)
        full_path = rel_path if os.path.isabs(rel_path) else os.path.join(IMAGES_ROOT, rel_path)
        image_paths.append(full_path)
    return image_paths


def load_benchmark(file_path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(items)} samples from {file_path}")
    return items


def ensure_results_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def flush_and_sync(*files: Any) -> None:
    for f in files:
        f.flush()
        os.fsync(f.fileno())


def build_prompt_part1(item: Dict[str, Any]) -> str:
    """Build prompt part 1: task instruction + article content."""
    content = item["content"]
    image_sequence = item["image_sequence"]
    title = item["title"]
    num_placeholders = item["num_placeholders"]

    return f"""## Task: Interleaved-Image-Text Matching

You are given an article about "{title}" with {num_placeholders} image placeholders marked as [IMAGE_PLACEHOLDER]. You are also given {len(image_sequence)} candidate images (Image 0, Image 1, ..., Image {len(image_sequence) - 1}) shown below.

Your task is to determine which image should be placed at each placeholder position based on the surrounding text context.

## Article Text (with placeholders):

{content}

## Candidate Images (Image 0 to Image {len(image_sequence) - 1}):
"""


def build_prompt_part2(item: Dict[str, Any]) -> str:
    """PROMPT 3 Part 2: detailed instruction section."""
    image_sequence = item["image_sequence"]
    num_placeholders = item["num_placeholders"]

    return f"""

## Instructions:

1. **Read the text carefully**: Each [IMAGE_PLACEHOLDER] appears within a specific context. The surrounding text describes what should be shown in that image.

2. **Analyze each placeholder**: For each placeholder (in order from first to last), identify what the nearby text is describing - this tells you what the image should show.

3. **Match images to placeholders**: Look at the {len(image_sequence)} candidate images provided and determine which image best matches the context around each placeholder.

4. **Important**: The same image index can only be used once. Each placeholder needs a different image.

## Output Format:

First reason step by step, then output your final answer on the LAST line as a Python list:
- Format: [{", ".join(["index" + str(i) for i in range(num_placeholders)])}]
- The list position corresponds to the placeholder order (first placeholder is index 0).
- Each value is the image index to place at that placeholder.
- Example: [2, 0, 1, 3, 4] means placeholder 1 uses Image 2, placeholder 2 uses Image 0, etc.
- Do NOT output the inverse mapping (i.e., image -> placeholder).
- The list must have exactly {num_placeholders} integers, each between 0 and {len(image_sequence) - 1}.

Now analyze the text and images, then provide your answer."""


def build_raw_input(item: Dict[str, Any]) -> Dict[str, Any]:
    image_paths = resolve_image_paths(item["image_sequence"])
    prompt_part1 = build_prompt_part1(item)
    prompt_part2 = build_prompt_part2(item)
    prompt_text = "\n".join(
        [prompt_part1] + [f"Image {i}:" for i in range(len(image_paths))] + [prompt_part2]
    )
    return {"prompt": prompt_text, "images": image_paths}

_LIST_RE = re.compile(r"\[([0-9,\s]+)\]")


def parse_prediction_list(text: str) -> Optional[List[int]]:
    if not text:
        return None
    matches = _LIST_RE.findall(text)
    if not matches:
        return None
    inner = matches[-1]
    parts = inner.replace(" ", "").split(",")
    try:
        return [int(p) for p in parts if p != ""]
    except ValueError:
        return None


def exact_match(pred: List[int], answer: List[int]) -> bool:
    return pred == answer


def partial_match(
    pred: List[int], answer: List[int], num_placeholders: Optional[int] = None
) -> float:
    return kendall_tau_mapped_0_1(pred, answer, num_placeholders)


def build_messages_for_vllm(
    item: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Build vLLM messages in text -> images -> text order.

    Message sequence:
    1. Text: prompt_part1 (task instruction + article content)
    2. Images: all candidate images (Image 0, Image 1, ...)
    3. Text: prompt_part2 (instruction details)

    Args:
        item: benchmark sample item
    """
    normalized = normalize_item_for_vllm(item)
    raw_input = build_raw_input(normalized)
    image_paths = raw_input["images"]
    prompt_part1 = build_prompt_part1(normalized)
    prompt_part2 = build_prompt_part2(normalized)

    content: List[Dict[str, Any]] = []
    content.append({"type": "text", "text": prompt_part1})

    # Add all images with explicit labels.
    for idx, full_path in enumerate(image_paths):
        content.append({"type": "text", "text": f"Image {idx}:"})
        # Provide both image/url fields for broader chat_template compatibility.
        content.append({"type": "image", "image": full_path, "url": full_path})

    content.append({"type": "text", "text": prompt_part2})
    return [{"role": "user", "content": content}], raw_input, normalized


def prepare_inputs_for_vllm(
    messages: List[Dict[str, Any]],
    processor: Any,
    model_path: str,
) -> Dict[str, Any]:
    """
    Prepare vLLM input with fallback strategy:
    - Prefer qwen_vl_utils for Qwen (keeps the known stable path)
    - Use the generic path by default for other models (e.g., GLM)
    - Automatically fall back when any path fails
    """
    errors: List[str] = []

    prefer_qwen_path = _is_qwen_model(model_path)
    if prefer_qwen_path:
        try:
            return _prepare_inputs_with_qwen_vision_utils(messages, processor)
        except Exception as err:
            errors.append(f"qwen_vision_utils: {type(err).__name__}: {err}")

    try:
        return _prepare_inputs_generic(messages, processor)
    except Exception as err:
        errors.append(f"generic: {type(err).__name__}: {err}")

    if not prefer_qwen_path:
        try:
            return _prepare_inputs_with_qwen_vision_utils(messages, processor)
        except Exception as err:
            errors.append(f"qwen_vision_utils_fallback: {type(err).__name__}: {err}")

    raise RuntimeError("prepare_inputs_for_vllm failed: " + " | ".join(errors))


def evaluate_vllm(
    model_path: str,
    benchmark_items: List[Dict[str, Any]],
    output_file: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_model_len: int = 16384,
    batch_size: int = 8,
    tensor_parallel_size: int = 1,
    disable_mm_preprocessor_cache: bool = True,
    max_prediction_retries: int = 10,
) -> Dict[str, Any]:
    """Run vLLM evaluation for a single benchmark file (batched inference without pre-check)."""
    from vllm import LLM, SamplingParams

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    base_llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "max_model_len": max_model_len,
    }
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be >= 1")
    base_llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
    if disable_mm_preprocessor_cache:
        # Keep compatibility across versions: newer vLLM uses mm_processor_cache_gb=0; older versions use disable flag.
        base_llm_kwargs["mm_processor_cache_gb"] = 0
        base_llm_kwargs["disable_mm_preprocessor_cache"] = True
        print("Stability mode: disabled vLLM multimodal preprocessor cache by default (to avoid mm_hash assertion).")

    processor = load_processor_with_compat(model_path)
    validate_glm_runtime_compat(model_path, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    total = len(benchmark_items)
    dropped_total = 0
    if max_prediction_retries < 0:
        raise ValueError("max_prediction_retries must be >= 0")
    max_prediction_attempts = max_prediction_retries + 1

    ensure_results_dir(output_file)
    resume_cleanup = prepare_output_file_for_resume(output_file)
    requeued_count = int(resume_cleanup.get("requeued_count", 0))
    resume_eval_info = load_existing_eval_results(output_file)
    evaluated_total = int(resume_eval_info["count"])
    exact_correct = float(resume_eval_info["exact_sum"])
    partial_score_sum = float(resume_eval_info["partial_sum"])
    null_prediction_count = int(resume_eval_info.get("null_count", 0))
    resumed = bool(evaluated_total > 0 or requeued_count > 0)

    completed_keys = set(resume_eval_info["keys"])
    skip_keys = completed_keys
    pending_items, duplicate_pending_count = collect_pending_items(benchmark_items, skip_keys)

    if requeued_count > 0:
        print(
            f"Detected {requeued_count} samples whose latest result is prediction=null; "
            f"removed from {output_file} and requeued."
        )
    if duplicate_pending_count > 0:
        print(
            f"Detected {duplicate_pending_count} duplicate data_id entries in pending samples; "
            "deduplicated automatically and kept the first occurrence only."
        )

    if resumed:
        print(
            f"Detected existing results: {evaluated_total}/{total} already evaluated, "
            f"historical prediction=null count={null_prediction_count} "
            f"(will be rerun automatically), remaining this run={len(pending_items)}."
        )

    if not pending_items:
        if evaluated_total <= 0:
            raise ValueError("No pending samples and historical results are empty.")
        exact_acc = exact_correct / evaluated_total
        partial_acc = partial_score_sum / evaluated_total
        print(
            f"No further inference needed. Valid samples={evaluated_total}, "
            f"exact_acc={exact_acc:.4f}, partial_acc={partial_acc:.4f}"
        )
        return {
            "mode": "vllm",
            "model_path": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "total": evaluated_total,
            "original_total": total,
            "exact_correct": int(exact_correct),
            "exact_accuracy": exact_acc,
            "partial_accuracy": partial_acc,
            "dropped_samples": 0,
            "output_file": output_file,
            "resumed": True,
        }

    with open(output_file, "a" if resumed else "w", encoding="utf-8") as out_f:

        def record_result(
            generated_text: str,
            item: Dict[str, Any],
            raw_input: Dict[str, Any],
            normalized: Dict[str, Any],
            attempts_used: int = 1,
            retry_errors: Optional[List[str]] = None,
        ) -> None:
            nonlocal exact_correct, partial_score_sum, evaluated_total
            answer = item["answer"]
            num_placeholders = normalized["num_placeholders"]
            pred_list = parse_prediction_list(generated_text)
            if pred_list is None:
                exact = 0.0
                partial = 0.0
            else:
                exact = 1.0 if exact_match(pred_list, answer) else 0.0
                partial = partial_match(pred_list, answer, num_placeholders)

            exact_correct += exact
            partial_score_sum += partial
            evaluated_total += 1

            model_input: Dict[str, Any] = {
                "prediction_retry_max": max_prediction_retries,
                "prediction_attempts": attempts_used,
            }
            if retry_errors:
                model_input["retry_errors"] = retry_errors
            if pred_list is None:
                model_input["error"] = "prediction_parse_failed"

            record = {
                "dataset_type": item["dataset_type"],
                "data_id": item["data_id"],
                "url_id": item["url_id"],
                "title": item["title"],
                "answer": answer,
                "prediction": pred_list,
                "raw_input": raw_input,
                "model_input": model_input,
                "raw_output": generated_text,
                "exact_correct": bool(exact == 1.0),
                "partial_score": partial,
                "partial_metric": "kendall_tau_0_1",
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[Stage-2] Inference started. Pending this run: {len(pending_items)} (total samples: {total}).")
        llm, resolved_llm_kwargs = build_llm_with_kwarg_compat(LLM, base_llm_kwargs)

        num_batches = (len(pending_items) + batch_size - 1) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(pending_items))
            batch_items = pending_items[start:end]

            batch_samples: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
            for item in batch_items:
                try:
                    messages, raw_input, normalized = build_messages_for_vllm(item)
                    sample_input = prepare_inputs_for_vllm(messages, processor, model_path)
                    batch_samples.append((item, raw_input, normalized, sample_input))
                except Exception as build_err:
                    dropped_total += 1
                    reason = (
                        "bad_image_inference" if is_bad_image_error(build_err) else "input_prepare_exception"
                    )
                    print(
                        f"[Stage-2] batch {batch_idx + 1}/{num_batches} input build failed; skipped 1 sample: "
                        f"reason={reason}, data_id={item.get('data_id')}, "
                        f"error={type(build_err).__name__}: {build_err}"
                    )

            if not batch_samples:
                print(
                    f"[Stage-2] Completed batch {batch_idx + 1}/{num_batches}, "
                    f"no inferable samples in this batch (total skipped: {dropped_total})"
                )
                continue

            batch_inputs = [sample[3] for sample in batch_samples]
            try:
                outputs = list(llm.generate(batch_inputs, sampling_params=sampling_params))
            except Exception as err:
                if is_mm_cache_assertion(err):
                    dropped_total += len(batch_samples)
                    print(
                        f"[Stage-2] batch {batch_idx + 1}/{num_batches} hit mm_hash assertion, "
                        f"skipping this batch ({len(batch_samples)} samples), rebuilding engine, then continuing."
                    )
                    try:
                        del llm
                        llm, _ = build_llm_with_kwarg_compat(LLM, resolved_llm_kwargs)
                    except Exception:
                        raise
                    continue
                raise

            sample_states: List[Dict[str, Any]] = []
            for sample_idx_in_batch, (item, raw_input, normalized, sample_input) in enumerate(batch_samples):
                generated_text = ""
                if sample_idx_in_batch < len(outputs):
                    output = outputs[sample_idx_in_batch]
                    if output is not None and getattr(output, "outputs", None):
                        generated_text = output.outputs[0].text or ""
                sample_states.append(
                    {
                        "item": item,
                        "raw_input": raw_input,
                        "normalized": normalized,
                        "sample_input": sample_input,
                        "generated_text": generated_text,
                        "attempts_used": 1,
                        "retry_errors": [],
                    }
                )

            pending_indices = [
                idx
                for idx, state in enumerate(sample_states)
                if parse_prediction_list(state["generated_text"]) is None
            ]
            if pending_indices:
                print(
                    f"[Stage-2] batch {batch_idx + 1}/{num_batches} after first pass has "
                    f"{len(pending_indices)}/{len(sample_states)} prediction=null samples; "
                    f"batch retries will run (max {max_prediction_retries} rounds)."
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
                    if is_mm_cache_assertion(retry_err):
                        print(
                            f"[Stage-2] batch {batch_idx + 1}/{num_batches} batch retry "
                            f"{attempt_idx}/{max_prediction_attempts} hit mm_hash assertion; "
                            "rebuilding engine before continuing."
                        )
                        err_msg = f"attempt_{attempt_idx}: mm_cache_assertion: {retry_err}"
                        try:
                            del llm
                            llm, _ = build_llm_with_kwarg_compat(LLM, resolved_llm_kwargs)
                        except Exception:
                            raise
                    else:
                        print(
                            f"[Stage-2] batch {batch_idx + 1}/{num_batches} batch retry "
                            f"{attempt_idx}/{max_prediction_attempts} failed: "
                            f"{type(retry_err).__name__}: {retry_err}"
                        )

                    for idx in pending_indices:
                        sample_states[idx]["retry_errors"].append(err_msg)
                    continue

                next_pending_indices: List[int] = []
                for output_idx, sample_idx in enumerate(pending_indices):
                    generated_text = ""
                    if output_idx < len(retry_outputs):
                        retry_output = retry_outputs[output_idx]
                        if retry_output is not None and getattr(retry_output, "outputs", None):
                            generated_text = retry_output.outputs[0].text or ""
                    sample_states[sample_idx]["generated_text"] = generated_text

                    if parse_prediction_list(generated_text) is None:
                        next_pending_indices.append(sample_idx)

                pending_indices = next_pending_indices

            for state in sample_states:
                record_result(
                    state["generated_text"],
                    state["item"],
                    state["raw_input"],
                    state["normalized"],
                    attempts_used=int(state["attempts_used"]),
                    retry_errors=list(state["retry_errors"]),
                )

            flush_and_sync(out_f)
            print(
                f"[Stage-2] Completed batch {batch_idx + 1}/{num_batches}, "
                f"processed {end}/{len(pending_items)} in total, skipped {dropped_total} in total"
            )

    if evaluated_total <= 0:
        raise ValueError("No valid sample left after inference")
    exact_acc = exact_correct / evaluated_total
    partial_acc = partial_score_sum / evaluated_total
    print(
        f"vLLM evaluation finished. Original samples={total}, valid samples={evaluated_total}, "
        f"skipped samples={dropped_total}, "
        f"exact_match correct={int(exact_correct)}, exact_acc={exact_acc:.4f}, "
        f"partial_acc={partial_acc:.4f}"
    )

    return {
        "mode": "vllm",
        "model_path": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "total": evaluated_total,
        "original_total": total,
        "exact_correct": int(exact_correct),
        "exact_accuracy": exact_acc,
        "partial_accuracy": partial_acc,
        "dropped_samples": dropped_total,
        "output_file": output_file,
        "resumed": resumed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Interleaved-Image-Text Matching vLLM evaluation script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to vLLM model")
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to benchmark jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output result jsonl")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_prediction_retries",
        type=int,
        default=10,
        help="Maximum retry rounds when prediction parses as None (excluding the first attempt)",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--disable_mm_preprocessor_cache",
        action="store_true",
        default=True,
        help=(
            "Enabled by default: disable vLLM multimodal preprocessor cache to avoid "
            "`Expected a cached item for mm_hash` assertion errors."
        ),
    )
    parser.add_argument(
        "--enable_mm_preprocessor_cache",
        dest="disable_mm_preprocessor_cache",
        action="store_false",
        help="Explicitly enable vLLM multimodal preprocessor cache (use only when your version is stable).",
    )

    args = parser.parse_args()
    random.seed(42)

    items = load_benchmark(args.benchmark_file)
    summary = evaluate_vllm(
        model_path=args.model_path,
        benchmark_items=items,
        output_file=args.output_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        max_prediction_retries=args.max_prediction_retries,
        tensor_parallel_size=args.tensor_parallel_size,
        disable_mm_preprocessor_cache=args.disable_mm_preprocessor_cache,
    )

    summary_path = args.output_file + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
