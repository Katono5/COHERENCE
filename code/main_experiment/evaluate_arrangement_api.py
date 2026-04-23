#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import io
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image
from tqdm import tqdm

try:
    from metrics import kendall_tau_mapped_0_1
except ImportError:
    from main_experiment.metrics import kendall_tau_mapped_0_1

# Image root directory (relative path)
IMAGES_ROOT = "../../datasets/images"
PLACEHOLDER = "[IMAGE_PLACEHOLDER]"


def count_lines(file_path: str) -> int:
    cnt = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def ensure_results_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


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


def load_existing_results(output_file: str) -> Dict[str, Any]:
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


def collect_pending_items(items: List[Dict[str, Any]], processed_keys: Set[str]) -> Tuple[List[Dict[str, Any]], int]:
    pending_items: List[Dict[str, Any]] = []
    seen_pending_keys: Set[str] = set()
    duplicate_count = 0
    for item in items:
        key = make_item_key(item)
        if key in processed_keys:
            continue
        if key in seen_pending_keys:
            duplicate_count += 1
            continue
        seen_pending_keys.add(key)
        pending_items.append(item)
    return pending_items, duplicate_count


def image_to_base64(image_path: str) -> str:
    """Read an image and encode it in base64."""
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=95)
            img_bytes = buffer.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        print(f"Warning: failed to read image {image_path}: {e}")
        return ""


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


def normalize_item_for_api(item: Dict[str, Any]) -> Dict[str, Any]:
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


def build_prompt_part1_v3(item: Dict[str, Any]) -> str:
    content = item["content"]
    image_sequence = item["image_sequence"]
    title = item.get("title", "")
    num_placeholders = item["num_placeholders"]

    return f"""## Task: Interleaved-Image-Text Matching

You are given an article about "{title}" with {num_placeholders} image placeholders marked as [IMAGE_PLACEHOLDER]. You are also given {len(image_sequence)} candidate images (Image 0, Image 1, ..., Image {len(image_sequence) - 1}) shown below.

Your task is to determine which image should be placed at each placeholder position based on the surrounding text context.

## Article Text (with placeholders):

{content}

## Candidate Images (Image 0 to Image {len(image_sequence) - 1}):
"""


def build_prompt_part2_v3(item: Dict[str, Any]) -> str:
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
    prompt_part1 = build_prompt_part1_v3(item)
    prompt_part2 = build_prompt_part2_v3(item)
    prompt_text = "\n".join(
        [prompt_part1] + [f"Image {i}:" for i in range(len(image_paths))] + [prompt_part2]
    )
    return {"prompt": prompt_text, "images": image_paths}


def _resolve_image_url(full_path: str, image_url_mode: str) -> Tuple[Optional[str], Optional[str]]:
    norm_path = os.path.normpath(full_path)
    if image_url_mode == "data_uri":
        b64 = image_to_base64(norm_path)
        if not b64:
            return None, f"image_encode_failed: {norm_path}"
        return f"data:image/jpeg;base64,{b64}", None
    if image_url_mode == "file_url":
        return f"file://{norm_path}", None
    if image_url_mode == "local_path":
        return norm_path, None
    return None, f"unknown_image_url_mode: {image_url_mode}"


def build_api_content_parts(
    item: Dict[str, Any],
    image_paths: List[str],
    image_url_mode: str = "data_uri",
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Build API input content in text(part1) -> images -> text(part2) order."""
    prompt_part1 = build_prompt_part1_v3(item)
    prompt_part2 = build_prompt_part2_v3(item)

    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt_part1}]

    for idx, full_path in enumerate(image_paths):
        image_url, err = _resolve_image_url(full_path, image_url_mode)
        if image_url is None:
            return [], err or "image_url_build_failed"
        content_parts.append({"type": "text", "text": f"Image {idx}:"})
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )

    content_parts.append({"type": "text", "text": prompt_part2})
    return content_parts, None



_LIST_RE = re.compile(r"\[([0-9,\s]+)\]")

def parse_prediction_list(text: str) -> Optional[List[int]]:
    """Parse [0,1,2,...] list from the last line of model output only."""
    if not text:
        return None

    stripped_text = text.rstrip()
    if not stripped_text:
        return None
    last_line = stripped_text.splitlines()[-1]

    matches = _LIST_RE.findall(last_line)
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


def partial_match(pred: List[int], answer: List[int], num_placeholders: Optional[int] = None) -> float:
    return kendall_tau_mapped_0_1(pred, answer, num_placeholders)


def _summarize_exception(exc: Exception, max_body_chars: int = 300) -> str:
    """Build a compact error string that keeps HTTP status/body when available."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    response = getattr(exc, "response", None)

    body_text: Optional[str] = None
    if body is not None:
        if isinstance(body, str):
            body_text = body
        else:
            try:
                body_text = json.dumps(body, ensure_ascii=False)
            except Exception:
                body_text = str(body)
    elif response is not None:
        body_text = getattr(response, "text", None)

    if isinstance(body_text, str):
        body_text = body_text.strip().replace("\n", " ")
        if len(body_text) > max_body_chars:
            body_text = body_text[:max_body_chars] + "...(truncated)"
    else:
        body_text = None

    parts = [f"{type(exc).__name__}: {exc}"]
    if status_code is not None:
        parts.insert(0, f"status={status_code}")
    if body_text:
        parts.append(f"body={body_text}")
    return " | ".join(parts)


def create_chat_completion_with_token_fallback(
    client: Any,
    req_kwargs: Dict[str, Any],
    max_tokens: int,
) -> Any:
    """
    Create chat completion with robust token-parameter fallback.

    Some OpenAI-compatible backends only support one of:
    - max_completion_tokens
    - max_tokens
    """
    try:
        return client.chat.completions.create(
            max_completion_tokens=max_tokens,
            **req_kwargs,
        )
    except Exception as first_error:
        first_summary = _summarize_exception(first_error)

    try:
        return client.chat.completions.create(
            max_tokens=max_tokens,
            **req_kwargs,
        )
    except Exception as second_error:
        second_summary = _summarize_exception(second_error)
        raise RuntimeError(
            "chat.completions.create failed with both token parameters; "
            f"first(max_completion_tokens): {first_summary}; "
            f"second(max_tokens): {second_summary}"
        ) from second_error


def evaluate_api(
    api_base: str,
    api_key: str,
    api_model: str,
    benchmark_file: str,
    output_file: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 8,
    image_url_mode: str = "auto",
    max_prediction_retries: int = 10,
) -> Dict[str, Any]:
    """Evaluate a multimodal model via an OpenAI-compatible API.

    Requests are sent concurrently, and the progress bar updates on each response.
    """
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)

    if image_url_mode not in {"auto", "data_uri", "file_url", "local_path"}:
        raise ValueError(f"invalid image_url_mode: {image_url_mode}")
    if max_prediction_retries < 0:
        raise ValueError("max_prediction_retries must be >= 0")
    if image_url_mode == "auto":
        image_url_modes = ["data_uri", "file_url", "local_path"]
    else:
        image_url_modes = [image_url_mode]
    max_prediction_attempts = max_prediction_retries + 1

    def create_chat_completion(messages: List[Dict[str, Any]]):
        req_kwargs: Dict[str, Any] = {
            "model": api_model,
            "messages": messages,
            "temperature": temperature,
        }
        return create_chat_completion_with_token_fallback(
            client=client,
            req_kwargs=req_kwargs,
            max_tokens=max_tokens,
        )

    def extract_message_content_and_reasoning(resp: Any) -> Tuple[str, Optional[Any]]:
        choices = getattr(resp, "choices", None)
        if not choices:
            raise ValueError("api response has no choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("api response choice has no message")

        content = getattr(message, "content", "")
        if content is None:
            generated_text = ""
        elif isinstance(content, str):
            generated_text = content
        else:
            generated_text = str(content)

        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is None:
            model_extra = getattr(message, "model_extra", None)
            if isinstance(model_extra, dict):
                reasoning_content = model_extra.get("reasoning_content")

        if reasoning_content is None and hasattr(resp, "model_dump"):
            try:
                resp_dict = resp.model_dump()
            except Exception:
                resp_dict = None
            if isinstance(resp_dict, dict):
                choices_list = resp_dict.get("choices")
                if isinstance(choices_list, list) and choices_list:
                    first_choice = choices_list[0]
                    if isinstance(first_choice, dict):
                        message_dict = first_choice.get("message")
                        if isinstance(message_dict, dict):
                            reasoning_content = message_dict.get("reasoning_content")

        return generated_text, reasoning_content

    # Read all samples first.
    items = []
    with open(benchmark_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    total = len(items)
    ensure_results_dir(output_file)
    resume_cleanup = prepare_output_file_for_resume(output_file)
    requeued_count = int(resume_cleanup.get("requeued_count", 0))
    resume_info = load_existing_results(output_file)
    exact_correct = float(resume_info["exact_sum"])
    partial_score_sum = float(resume_info["partial_sum"])
    processed_keys = set(resume_info["keys"])
    processed_count = int(resume_info["count"])
    null_prediction_count = int(resume_info.get("null_count", 0))
    miss_prediction_count = null_prediction_count
    resumed = processed_count > 0
    pending_items, duplicate_pending_count = collect_pending_items(items, processed_keys)

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
            f"Detected existing results in {output_file}: completed {processed_count}/{total}, "
            f"prediction=null count={null_prediction_count}, "
            f"continuing with remaining {len(pending_items)} samples."
        )

    if not pending_items:
        exact_acc = exact_correct / total if total > 0 else 0.0
        partial_acc = partial_score_sum / total if total > 0 else 0.0
        miss_acc = miss_prediction_count / total if total > 0 else 0.0
        print(
            f"No further inference needed. Total samples={total}, "
            f"exact_acc={exact_acc:.4f}, partial_acc={partial_acc:.4f}, "
            f"miss(prediction=null)={miss_prediction_count}, miss_acc={miss_acc:.4f}"
        )
        return {
            "mode": "api",
            "total": total,
            "exact_correct": int(exact_correct),
            "exact_accuracy": exact_acc,
            "partial_accuracy": partial_acc,
            "miss_prediction_null": miss_prediction_count,
            "miss_accuracy": miss_acc,
            "output_file": output_file,
            "resumed": True,
        }

    out_f = open(output_file, "a" if resumed else "w", encoding="utf-8")
    write_lock = threading.Lock()
    print(
        f"API evaluation started. Total samples={total}, pending this run={len(pending_items)}, "
        f"concurrency={batch_size}, image_url_mode={image_url_mode}"
    )

    def process_single_item(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process one sample and return its result record."""
        model_input_for_log: Optional[Dict[str, Any]] = None

        try:
            normalized = normalize_item_for_api(item)
        except Exception as e:
            generated_text = ""
            answer = item.get("answer", [])
            num_placeholders = len(answer) if isinstance(answer, list) else 0
            record = {
                "idx": idx,
                "dataset_type": item.get("dataset_type", ""),
                "data_id": item.get("data_id", ""),
                "url_id": item.get("url_id", ""),
                "title": item.get("title", ""),
                "answer": answer,
                "prediction": None,
                "raw_input": {"error": f"normalize_failed: {e}"},
                "model_input": {"error": f"normalize_failed: {e}"},
                "raw_output": generated_text,
                "reasoning_content": None,
                "exact_correct": False,
                "partial_score": 0.0,
                "partial_metric": "kendall_tau_0_1",
            }
            return record

        raw_input = build_raw_input(normalized)
        image_paths = raw_input["images"]
        generated_text = ""
        reasoning_content: Optional[Any] = None
        pred_list: Optional[List[int]] = None
        chosen_image_url_mode: Optional[str] = None
        api_errors: List[str] = []
        redacted_content: List[Dict[str, Any]] = []
        attempts_used = 0

        for attempt_idx in range(1, max_prediction_attempts + 1):
            attempts_used = attempt_idx
            generated_text = ""
            reasoning_content = None
            chosen_image_url_mode = None
            redacted_content_in_attempt: List[Dict[str, Any]] = []

            for mode in image_url_modes:
                content_parts, content_err = build_api_content_parts(
                    normalized,
                    image_paths,
                    image_url_mode=mode,
                )
                if content_err is not None:
                    api_errors.append(f"attempt_{attempt_idx}:{mode}: {content_err}")
                    continue

                redacted_content_in_attempt = []
                for part in content_parts:
                    if part.get("type") == "image_url":
                        redacted_content_in_attempt.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": "[image]"},
                            }
                        )
                    else:
                        redacted_content_in_attempt.append(part)

                messages = [{"role": "user", "content": content_parts}]

                try:
                    resp = create_chat_completion(messages)
                    generated_text, reasoning_content = extract_message_content_and_reasoning(resp)
                    chosen_image_url_mode = mode
                    break
                except Exception:
                    api_errors.append(
                        f"attempt_{attempt_idx}:{mode}: request_failed"
                    )

            if redacted_content_in_attempt:
                redacted_content = redacted_content_in_attempt

            pred_list = parse_prediction_list(generated_text)
            if pred_list is not None:
                break

        model_input_for_log = {
            "raw_input": raw_input,
            "messages": [{"role": "user", "content": redacted_content}] if redacted_content else None,
            "image_url_modes_tried": image_url_modes,
            "chosen_image_url_mode": chosen_image_url_mode,
            "prediction_retry_max": max_prediction_retries,
            "prediction_attempts": attempts_used,
        }
        if api_errors:
            model_input_for_log["api_errors"] = api_errors
            if pred_list is None:
                model_input_for_log["error"] = "api_request_failed_or_prediction_parse_failed"

        answer = item.get("answer", [])
        num_placeholders = normalized["num_placeholders"]
        if num_placeholders <= 0 and isinstance(answer, list):
            num_placeholders = len(answer)
        if pred_list is None:
            exact = 0.0
            partial = 0.0
        else:
            exact = 1.0 if exact_match(pred_list, answer) else 0.0
            partial = partial_match(pred_list, answer, num_placeholders)

        record = {
            "idx": idx,  # Used for ordered output writing.
            "dataset_type": item.get("dataset_type", ""),
            "data_id": item.get("data_id", ""),
            "url_id": item.get("url_id", ""),
            "title": item.get("title", ""),
            "answer": answer,
            "prediction": pred_list,
            "raw_input": model_input_for_log.get("raw_input") if isinstance(model_input_for_log, dict) else None,
            "model_input": model_input_for_log,
            "input_messages": model_input_for_log.get("messages") if isinstance(model_input_for_log, dict) else None,
            "api_error": model_input_for_log.get("api_errors") if isinstance(model_input_for_log, dict) else None,
            "raw_output": generated_text,
            "reasoning_content": reasoning_content,
            "exact_correct": bool(exact == 1.0),
            "partial_score": partial,
            "partial_metric": "kendall_tau_0_1",
        }
        return record

    # Buffer for ordered writes.
    results_buffer = {}
    next_write_idx = 0
    evaluated_count = processed_count

    def try_write_ordered():
        """Write completed results in index order when possible."""
        nonlocal next_write_idx
        while next_write_idx in results_buffer:
            record = results_buffer.pop(next_write_idx)
            # Remove the internal index field used for ordering.
            record.pop("idx")
            # Write to file.
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            next_write_idx += 1

    # Process concurrently and update progress after each completion.
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {executor.submit(process_single_item, idx, item): idx
                   for idx, item in enumerate(pending_items)}
        
        with tqdm(total=len(pending_items), desc="API inference") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    record = future.result()
                    with write_lock:
                        exact_correct += 1.0 if record["exact_correct"] else 0.0
                        partial_score_sum += record["partial_score"]
                        if record.get("prediction") is None:
                            miss_prediction_count += 1
                        evaluated_count += 1
                        results_buffer[record["idx"]] = record
                        try_write_ordered()
                except Exception as e:
                    print(f"Error while processing sample idx={idx}: {e}")
                pbar.update(1)
                current_done = evaluated_count
                live_exact_acc = exact_correct / current_done if current_done > 0 else 0.0
                live_partial_acc = partial_score_sum / current_done if current_done > 0 else 0.0
                live_miss_acc = miss_prediction_count / current_done if current_done > 0 else 0.0
                pbar.set_postfix(
                    exact_acc=f"{live_exact_acc:.4f}",
                    partial_acc=f"{live_partial_acc:.4f}",
                    miss_null=f"{miss_prediction_count}/{current_done}({live_miss_acc:.4f})",
                )

    out_f.close()

    exact_acc = exact_correct / total if total > 0 else 0.0
    partial_acc = partial_score_sum / total if total > 0 else 0.0
    miss_acc = miss_prediction_count / total if total > 0 else 0.0
    print(
        f"API evaluation finished. Total samples={total}, "
        f"exact_match correct={int(exact_correct)}, exact_acc={exact_acc:.4f}, "
        f"partial_acc={partial_acc:.4f}, "
        f"miss(prediction=null)={miss_prediction_count}, miss_acc={miss_acc:.4f}"
    )

    return {
        "mode": "api",
        "total": total,
        "exact_correct": int(exact_correct),
        "exact_accuracy": exact_acc,
        "partial_accuracy": partial_acc,
        "miss_prediction_null": miss_prediction_count,
        "miss_accuracy": miss_acc,
        "output_file": output_file,
        "resumed": resumed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="category-1-arrangement API evaluation script")
    parser.add_argument("--api_base", type=str, required=True, help="API base_url")
    parser.add_argument("--api_key", type=str, required=True, help="API key")
    parser.add_argument("--api_model", type=str, required=True, help="API model name")
    parser.add_argument("--benchmark_file", type=str, required=True, help="Path to benchmark jsonl file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output result jsonl")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of concurrent requests")
    parser.add_argument(
        "--max_prediction_retries",
        type=int,
        default=10,
        help="Maximum retry rounds when prediction parses as None (excluding the first attempt)",
    )
    parser.add_argument(
        "--image_url_mode",
        type=str,
        default="auto",
        choices=["auto", "data_uri", "file_url", "local_path"],
        help="Image URL transport mode; auto tries data_uri/file_url/local_path in order",
    )

    args = parser.parse_args()
    random.seed(42)

    summary = evaluate_api(
        api_base=args.api_base,
        api_key=args.api_key,
        api_model=args.api_model,
        benchmark_file=args.benchmark_file,
        output_file=args.output_file,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        image_url_mode=args.image_url_mode,
        max_prediction_retries=args.max_prediction_retries,
    )

    summary_path = args.output_file + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
