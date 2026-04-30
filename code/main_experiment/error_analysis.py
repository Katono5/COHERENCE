#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **_kwargs):
        return iterable


IMAGES_ROOT = "../../datasets/images"

SUBSETS = ["wikihow", "storybird", "cooking", "science"]
RESULT_FILE_RE = re.compile(
    r"^(?P<subset>wikihow|storybird|cooking|science)_full\.reasonable_[^/]+_eval\.jsonl$"
)
LEGACY_RESULT_FILE_RE = re.compile(
    r"^(?P<model>.+)_(?P<subset>wikihow|storybird|cooking|science)_full\.reasonable_[^/]+_eval\.jsonl$"
)

ERROR_TYPES = [
    "Global Assignment Drift",
    "Step-State Confusion",
    "Fine-Detail Miss",
    "Semantic Over-Interpretation",
    "Visual Hallucination",
    "Instruction Violation",
]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSON object expected at {path}:{line_no}")
            yield obj


def parse_result_subset(filename: str) -> Optional[str]:
    matched = RESULT_FILE_RE.match(filename)
    if matched:
        return matched.group("subset")
    legacy = LEGACY_RESULT_FILE_RE.match(filename)
    if legacy:
        return legacy.group("subset")
    return None


def parse_result_model_subset(filename: str, parent_dir: str) -> Optional[Tuple[str, str]]:
    legacy = LEGACY_RESULT_FILE_RE.match(filename)
    if legacy:
        return legacy.group("model"), legacy.group("subset")

    matched = RESULT_FILE_RE.match(filename)
    if matched:
        model_name = os.path.basename(os.path.normpath(parent_dir)) or "model"
        return model_name, matched.group("subset")

    return None


def discover_result_files(result_dir: str) -> List[Dict[str, str]]:
    discovered: List[Dict[str, str]] = []
    for current_root, dir_names, file_names in os.walk(result_dir):
        dir_names.sort()
        for filename in sorted(file_names):
            if not filename.endswith(".jsonl"):
                continue
            parsed = parse_result_model_subset(filename, parent_dir=current_root)
            if parsed is None:
                continue
            model_name, subset = parsed
            discovered.append(
                {
                    "model": model_name,
                    "subset": subset,
                    "path": os.path.join(current_root, filename),
                }
            )

    if not discovered:
        raise ValueError(
            "No valid eval jsonl files found under "
            f"{result_dir}. Expected names like "
            "'model_wikihow_full.reasonable_vllm_eval.jsonl' or "
            "'wikihow_full.reasonable_vllm_eval.jsonl'."
        )

    discovered.sort(key=lambda item: (item["model"], item["subset"], item["path"]))
    return discovered


def find_model_result_files(result_dir: str) -> Dict[str, str]:
    discovered = discover_result_files(result_dir)
    matched: Dict[str, str] = {}
    for item in discovered:
        subset = item["subset"]
        path = item["path"]
        if subset in matched and matched[subset] != path:
            raise ValueError(
                "Multiple result files found for subset "
                f"'{subset}' under {result_dir}. Please use discover_result_files()."
            )
        matched[subset] = path

    missing = [subset for subset in SUBSETS if subset not in matched]
    if missing:
        raise ValueError(f"Input folder {result_dir} is missing subset files: {', '.join(missing)}")

    return matched


def parse_exact_correct(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:
            return None
        return float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def is_wrong_case(record: Dict[str, Any]) -> bool:
    exact = parse_exact_correct(record.get("exact_correct"))
    if exact is not None:
        return not exact

    prediction = record.get("prediction")
    answer = record.get("answer")
    if isinstance(prediction, list) and isinstance(answer, list):
        return prediction != answer

    return True


def load_wrong_cases(
    result_files: List[Dict[str, str]],
    max_cases: Optional[int],
    max_candidate_images: Optional[int] = None,
) -> List[Dict[str, Any]]:
    wrong_cases: List[Dict[str, Any]] = []
    for item in result_files:
        subset = item["subset"]
        path = item["path"]
        model_name = item.get("model", "model")
        for obj in iter_jsonl(path):
            if not is_wrong_case(obj):
                continue
            copied = dict(obj)
            copied["subset"] = subset
            copied["source_result_file"] = path
            copied["model_name"] = model_name
            copied["candidate_image_count"] = extract_candidate_image_count(copied)
            if max_candidate_images is not None:
                image_count = copied["candidate_image_count"]
                if image_count is None or image_count > max_candidate_images:
                    continue
            wrong_cases.append(copied)
            if max_cases is not None and len(wrong_cases) >= max_cases:
                return wrong_cases
    return wrong_cases


def extract_raw_input_payload(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_input = record.get("raw_input")
    if isinstance(raw_input, dict):
        return raw_input

    model_input = record.get("model_input")
    if isinstance(model_input, dict):
        nested_raw_input = model_input.get("raw_input")
        if isinstance(nested_raw_input, dict):
            return nested_raw_input

    return None


def extract_original_problem_input(record: Dict[str, Any]) -> Tuple[str, List[str]]:
    raw_input = extract_raw_input_payload(record)
    if raw_input is None:
        return "", []

    prompt = raw_input.get("prompt")
    prompt_text = prompt if isinstance(prompt, str) else ""

    images = raw_input.get("images")
    if isinstance(images, list):
        image_list = [str(item) for item in images]
    else:
        image_list = []

    return prompt_text, image_list


def extract_candidate_image_count(record: Dict[str, Any]) -> Optional[int]:
    _, image_list = extract_original_problem_input(record)
    if image_list:
        return len(image_list)

    answer = record.get("answer")
    if isinstance(answer, list):
        return len(answer)

    prediction = record.get("prediction")
    if isinstance(prediction, list):
        return len(prediction)

    return None


def extract_model_raw_output(record: Dict[str, Any]) -> Any:
    if "raw_output" in record:
        return record.get("raw_output")

    model_input = record.get("model_input")
    if isinstance(model_input, dict) and "raw_output" in model_input:
        return model_input.get("raw_output")

    return None


def extract_candidate_images(record: Dict[str, Any]) -> List[str]:
    _, image_list = extract_original_problem_input(record)
    return image_list


def resolve_image_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(IMAGES_ROOT, path)


def build_judge_text(record: Dict[str, Any]) -> str:
    prompt_text, _ = extract_original_problem_input(record)
    raw_output = extract_model_raw_output(record)
    if raw_output is None:
        raw_output = ""
    raw_output = str(raw_output)
    prediction = record.get("prediction")
    answer = record.get("answer")

    return f"""You are an error analyst for interleaved image-text matching.

Your task is to perform systematic root-cause analysis for an incorrect prediction,
then provide:
1) one paragraph explaining why the prediction is wrong,
2) one primary error type,
3) two secondary error types.
This is a long-context interleaved image-text assignment task.
You must jointly evaluate text evidence, image evidence, and structural constraints.
Do not judge from local fragments only.

Use exactly one label for primary_error_type and exactly two labels for secondary_error_types.
Allowed labels (exact string match) and interpretation criteria:
- Global Assignment Drift: Local image-text pairings may look reasonable, but the final mapping is globally inconsistent across the full article (e.g., systematic shift, wrong overall alignment, boundary drift).
- Step-State Confusion: The mismatch mainly comes from mixing up nearby steps/states that are semantically close or visually similar.
- Fine-Detail Miss: The mismatch is caused by missing decisive fine-grained cues (small objects, subtle state changes, local attributes, tool/action details).
- Semantic Over-Interpretation: The mismatch is driven by reading more meaning into an image than the visible evidence supports, then forcing that interpretation into alignment.
- Visual Hallucination: The reasoning relies on visual elements that are not actually present in the image.
- Instruction Violation: The output breaks task constraints or format requirements (e.g., invalid list format, duplicate image index use, missing/extra assignments, illegal indices).

Input:
[Original Prompt]
{prompt_text}

[Model Output]
{raw_output}

[Prediction]
{json.dumps(prediction, ensure_ascii=False)}

[Gold Answer]
{json.dumps(answer, ensure_ascii=False)}

Systematic analysis protocol (must follow internally before choosing labels):
1. Constraint & format verification:
   - Check task-structure violations (length mismatch, illegal indices, repeated image use, invalid format).
   - If structural non-compliance is the dominant issue, prioritize Instruction Violation.
2. Global assignment consistency:
   - Evaluate whether the global one-to-one mapping across all placeholders and all candidate images is coherent.
   - If local matches look plausible but the full assignment drifts (global mismatch or boundary shift), prioritize Global Assignment Drift.
3. Local step/state discrimination:
   - Compare adjacent steps, similar states, and visually similar moments.
   - If the main issue is confusion between nearby stages, prioritize Step-State Confusion.
4. Fine-grained evidence grounding:
   - Check whether key detail cues are missed (tools, action details, state transitions, local attributes).
   - If missed fine-grained evidence drives the error, prioritize Fine-Detail Miss.
5. Semantic calibration:
   - Check whether semantic interpretation exceeds available evidence.
   - If over-interpretation drives mismatch, prioritize Semantic Over-Interpretation.
6. Visual fidelity:
   - Check whether the model relies on objects/attributes/relations not present in the image.
   - If fabricated visual evidence is used, prioritize Visual Hallucination.
7. Write reason paragraph and choose labels:
   - Write error_reason as one concise paragraph (3-6 sentences) that directly explains the observed mismatch.
   - The paragraph must refer to concrete mismatch patterns in this sample (not generic comments).
   - Select primary_error_type as the single dominant cause of the wrong assignment.
   - Select two secondary_error_types that are relevant but secondary; no duplicates and no overlap with primary.

Return exactly one JSON object (no markdown, no explanation text):
{{
  "error_reason": "one concise paragraph explaining why the prediction is wrong",
  "primary_error_type": "one label from the six",
  "secondary_error_types": ["one label", "one label"]
}}

"""


def build_judge_messages(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": build_judge_text(record)}]
    image_paths = extract_candidate_images(record)
    for idx, image_path in enumerate(image_paths):
        full_path = resolve_image_path(image_path)
        content.append({"type": "text", "text": f"Image {idx}:"})
        content.append({"type": "image", "image": full_path, "url": full_path})
    return [{"role": "user", "content": content}]


def extract_first_json_object(raw_output: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw_output):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw_output[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No JSON object found in judge output")


def parse_judge_output_strict(raw_output: str) -> Dict[str, Any]:
    obj = extract_first_json_object(raw_output)

    error_reason = obj.get("error_reason")
    primary = obj.get("primary_error_type")
    secondary = obj.get("secondary_error_types")

    if not isinstance(error_reason, str) or not error_reason.strip():
        raise ValueError(f"Invalid error_reason: {error_reason}")

    if not isinstance(primary, str) or primary not in ERROR_TYPES:
        raise ValueError(f"Invalid primary_error_type: {primary}")

    if not isinstance(secondary, list) or len(secondary) != 2:
        raise ValueError(f"Invalid secondary_error_types length: {secondary}")

    if not all(isinstance(item, str) for item in secondary):
        raise ValueError(f"secondary_error_types must be list[str]: {secondary}")

    if len(set(secondary)) != 2:
        raise ValueError(f"secondary_error_types has duplicates: {secondary}")

    if primary in secondary:
        raise ValueError(
            f"secondary_error_types must not contain primary_error_type: {primary}, {secondary}"
        )

    if any(item not in ERROR_TYPES for item in secondary):
        raise ValueError(f"secondary_error_types contains unknown labels: {secondary}")

    return {
        "error_reason": error_reason.strip(),
        "primary_error_type": primary,
        "secondary_error_types": secondary,
    }


def parse_judge_output_with_retry(
    initial_raw_output: str,
    max_retries: int,
    regenerate_output: Optional[Callable[[], str]] = None,
) -> Tuple[Dict[str, Any], str, int]:
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    raw_output = str(initial_raw_output or "")
    retries_used = 0

    while True:
        try:
            parsed = parse_judge_output_strict(raw_output)
            return parsed, raw_output, retries_used
        except ValueError as exc:
            if retries_used >= max_retries or regenerate_output is None:
                preview = raw_output.strip().replace("\n", "\\n")
                if len(preview) > 200:
                    preview = preview[:200] + "...(truncated)"
                raise ValueError(
                    "Failed to parse judge output JSON after "
                    f"{retries_used} retries. Last error: {exc}. "
                    f"Raw preview: {preview}"
                ) from exc
            retries_used += 1
            regenerated = regenerate_output()
            raw_output = "" if regenerated is None else str(regenerated)


def load_processor(model_path: str) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def collect_pil_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
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


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = collect_pil_images_from_messages(messages)
    if not image_inputs:
        raise ValueError("No candidate images found in record.raw_input.images")
    return {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs},
    }


def create_vllm_runner(
    model_path: str,
    tensor_parallel_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[Any, Any, Any]:
    from vllm import LLM, SamplingParams

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(1, int(tensor_parallel_size)),
        max_model_len=max(1, int(max_model_len)),
    )
    sampling_params = SamplingParams(
        max_tokens=max(1, int(max_new_tokens)),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    processor = load_processor(model_path)
    return llm, sampling_params, processor


def extract_generated_text(output_obj: Any) -> str:
    outputs = getattr(output_obj, "outputs", None)
    if not outputs:
        return ""
    first = outputs[0]
    text = getattr(first, "text", "")
    return str(text or "")


def infer_model_name(result_dir: str) -> str:
    return os.path.basename(os.path.normpath(result_dir)) or "model"


def default_output_paths(result_dir: str) -> Tuple[str, str]:
    output_dir = f"{os.path.normpath(result_dir)}_error_analysis"
    model_name = infer_model_name(result_dir)
    jsonl_path = os.path.join(output_dir, f"{model_name}_analysis.jsonl")
    summary_path = os.path.join(output_dir, f"{model_name}_analysis.summary.json")
    return jsonl_path, summary_path


def write_summary(path: str, summary: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def normalize_error_type_to_filename(error_type: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(error_type).strip().lower()).strip("_")
    return f"{slug or 'unknown_error_type'}.jsonl"


def export_grouped_by_primary_error_type(
    reviewed_records: List[Dict[str, Any]],
    output_dir: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in reviewed_records:
        label = str(record.get("primary_error_type") or "Unknown")
        grouped.setdefault(label, []).append(record)

    ordered_labels = [label for label in ERROR_TYPES if label in grouped]
    extra_labels = sorted(label for label in grouped.keys() if label not in ERROR_TYPES)
    labels_in_order = ordered_labels + extra_labels

    files: Dict[str, str] = {}
    combined_records: List[Dict[str, Any]] = []
    by_primary_error_type: Dict[str, int] = {}

    for label in labels_in_order:
        records = grouped[label]
        by_primary_error_type[label] = len(records)
        file_path = os.path.join(output_dir, normalize_error_type_to_filename(label))
        with open(file_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        files[label] = file_path
        combined_records.extend(records)

    combined_path = os.path.join(
        output_dir,
        f"combined_primary_grouped_{len(combined_records)}.jsonl",
    )
    with open(combined_path, "w", encoding="utf-8") as f:
        for record in combined_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = os.path.join(output_dir, "summary.json")
    write_summary(
        summary_path,
        {
            "output_dir": output_dir,
            "total_records": len(combined_records),
            "by_primary_error_type": by_primary_error_type,
            "files": files,
            "combined_file": combined_path,
        },
    )
    return {
        "output_dir": output_dir,
        "total_records": len(combined_records),
        "by_primary_error_type": by_primary_error_type,
        "files": files,
        "combined_file": combined_path,
        "summary_file": summary_path,
    }


def build_reviewed_record(
    case: Dict[str, Any], parsed: Dict[str, Any], raw_judge_output: str
) -> Dict[str, Any]:
    original_input_text, original_input_images = extract_original_problem_input(case)
    model_raw_output = extract_model_raw_output(case)

    reviewed = {
        "model_name": case.get("model_name"),
        "subset": case.get("subset"),
        "candidate_image_count": case.get("candidate_image_count"),
        "source_result_file": case.get("source_result_file"),
        "dataset_type": case.get("dataset_type"),
        "data_id": case.get("data_id"),
        "url_id": case.get("url_id"),
        "title": case.get("title"),
        "answer": case.get("answer"),
        "prediction": case.get("prediction"),
        "model_raw_input": extract_raw_input_payload(case),
        "model_prediction": case.get("prediction"),
        "model_raw_output": model_raw_output,
        "original_input_text_with_placeholders": original_input_text,
        "original_input_image_list": original_input_images,
        "judge_error_reason": parsed["error_reason"],
        "primary_error_type": parsed["primary_error_type"],
        "secondary_error_types": parsed["secondary_error_types"],
        "error_type_classification": {
            "primary_error_type": parsed["primary_error_type"],
            "secondary_error_types": parsed["secondary_error_types"],
        },
        "judge_output_parsed": parsed,
        "judge_model_output": raw_judge_output,
        "judge_raw_output": raw_judge_output,
    }
    return reviewed


def analyze_errors_with_vllm(
    result_dir: str,
    vllm_model_path: str,
    output_jsonl: Optional[str],
    output_summary: Optional[str],
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_cases: Optional[int],
    max_candidate_images: Optional[int],
    max_parse_retries: int,
    parse_error_policy: str,
) -> Dict[str, Any]:
    if max_candidate_images is not None and max_candidate_images <= 0:
        max_candidate_images = None
    if max_parse_retries < 0:
        raise ValueError(f"max_parse_retries must be >= 0, got {max_parse_retries}")
    if parse_error_policy not in {"skip", "raise"}:
        raise ValueError(
            f"parse_error_policy must be one of ['skip', 'raise'], got {parse_error_policy}"
        )

    result_files = discover_result_files(result_dir)
    wrong_cases = load_wrong_cases(
        result_files=result_files,
        max_cases=max_cases,
        max_candidate_images=max_candidate_images,
    )

    if output_jsonl is None or output_summary is None:
        default_jsonl, default_summary = default_output_paths(result_dir)
        if output_jsonl is None:
            output_jsonl = default_jsonl
        if output_summary is None:
            output_summary = default_summary

    ensure_parent_dir(output_jsonl)

    llm, sampling_params, processor = create_vllm_runner(
        model_path=vllm_model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    reviewed_records: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []
    total_parse_retries_used = 0
    step = max(1, int(batch_size))

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(wrong_cases), step), desc=f"error_analysis(vllm_batch={step})"):
            batch = wrong_cases[start : start + step]

            prepared_inputs: List[Dict[str, Any]] = []
            batch_cases: List[Dict[str, Any]] = []
            for case in batch:
                messages = build_judge_messages(case)
                prepared_input = prepare_inputs_for_vllm(messages, processor)
                prepared_inputs.append(prepared_input)
                batch_cases.append(case)

            outputs = list(llm.generate(prepared_inputs, sampling_params=sampling_params))
            if len(outputs) != len(batch_cases):
                raise RuntimeError(
                    "vLLM output count mismatch: "
                    f"expected={len(batch_cases)}, got={len(outputs)}"
                )

            for case, prepared_input, output_obj in zip(batch_cases, prepared_inputs, outputs):
                initial_raw_judge_output = extract_generated_text(output_obj)

                def regenerate_once() -> str:
                    retry_outputs = list(llm.generate([prepared_input], sampling_params=sampling_params))
                    if len(retry_outputs) != 1:
                        raise RuntimeError(
                            "vLLM retry output count mismatch: "
                            f"expected=1, got={len(retry_outputs)}"
                        )
                    return extract_generated_text(retry_outputs[0])

                try:
                    parsed, final_raw_judge_output, retries_used = parse_judge_output_with_retry(
                        initial_raw_output=initial_raw_judge_output,
                        max_retries=max_parse_retries,
                        regenerate_output=regenerate_once if max_parse_retries > 0 else None,
                    )
                    total_parse_retries_used += retries_used
                except ValueError as exc:
                    if parse_error_policy == "raise":
                        raise
                    case_id = (
                        case.get("data_id")
                        or case.get("url_id")
                        or case.get("title")
                        or f"offset_{start}"
                    )
                    parse_failures.append(
                        {
                            "subset": case.get("subset"),
                            "case_id": str(case_id),
                            "error": str(exc),
                        }
                    )
                    print(
                        "[parse_error][skip] "
                        f"subset={case.get('subset')} case_id={case_id} err={exc}"
                    )
                    continue

                reviewed = build_reviewed_record(case, parsed, final_raw_judge_output)
                reviewed_records.append(reviewed)
                out_f.write(json.dumps(reviewed, ensure_ascii=False) + "\n")

    by_primary_error_type = Counter(
        str(record["primary_error_type"]) for record in reviewed_records
    )
    by_secondary_error_type = Counter()
    for record in reviewed_records:
        for label in record["secondary_error_types"]:
            by_secondary_error_type[str(label)] += 1

    grouped_primary_dir = os.path.join(
        os.path.dirname(output_jsonl),
        "collected_samples_by_primary_type",
    )
    grouped_primary_summary = export_grouped_by_primary_error_type(
        reviewed_records=reviewed_records,
        output_dir=grouped_primary_dir,
    )

    summary = {
        "result_dir": result_dir,
        "vllm_model_path": vllm_model_path,
        "matched_result_files": result_files,
        "matched_result_file_count": len(result_files),
        "wrong_case_total_count": len(wrong_cases),
        "max_candidate_images": max_candidate_images,
        "total_reviewed": len(reviewed_records),
        "parse_failures_skipped": len(parse_failures),
        "parse_failures_examples": parse_failures[:20],
        "max_parse_retries": max_parse_retries,
        "parse_retries_used_total": total_parse_retries_used,
        "parse_error_policy": parse_error_policy,
        "max_cases": max_cases,
        "output_jsonl": output_jsonl,
        "output_summary": output_summary,
        "by_primary_error_type": dict(by_primary_error_type),
        "by_secondary_error_type": dict(by_secondary_error_type),
        "grouped_by_primary_error_type_output": grouped_primary_summary,
        "error_types": ERROR_TYPES,
    }
    write_summary(output_summary, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "vLLM error analysis for interleaved multimodal matching. Supports "
            "single-model folders and aggregated results roots."
        )
    )
    parser.add_argument("--result_dir", type=str, required=True, help="Folder of eval jsonl files.")
    parser.add_argument(
        "--vllm_model_path",
        type=str,
        default="<VLLM_MODEL_PATH>",
        help="vLLM model path for judge inference.",
    )
    parser.add_argument("--output_jsonl", type=str, default=None, help="Per-sample output jsonl.")
    parser.add_argument("--output_summary", type=str, default=None, help="Summary json path.")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="vLLM tensor parallel size.")
    parser.add_argument("--batch_size", type=int, default=128, help="vLLM batch size.")
    parser.add_argument("--max_model_len", type=int, default=30000, help="vLLM max_model_len.")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="vLLM max tokens.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Sampling top_p.")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional cap on wrong cases.")
    parser.add_argument(
        "--max_candidate_images",
        type=int,
        default=0,
        help=(
            "Only analyze wrong cases whose candidate-image count is <= this value. "
            "Set <=0 to disable the filter."
        ),
    )
    parser.add_argument(
        "--max_parse_retries",
        type=int,
        default=1,
        help="Retry count when judge output cannot be parsed as valid JSON.",
    )
    parser.add_argument(
        "--parse_error_policy",
        type=str,
        choices=["skip", "raise"],
        default="skip",
        help="On parse failure after retries: skip the case or raise an exception.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result_dir = os.path.normpath(args.result_dir)
    if not os.path.isdir(result_dir):
        raise ValueError(f"result_dir does not exist or is not a directory: {result_dir}")
    if not args.vllm_model_path or args.vllm_model_path == "<VLLM_MODEL_PATH>":
        raise ValueError("Please provide --vllm_model_path.")

    max_candidate_images = args.max_candidate_images
    if max_candidate_images is not None and max_candidate_images <= 0:
        max_candidate_images = None

    summary = analyze_errors_with_vllm(
        result_dir=result_dir,
        vllm_model_path=args.vllm_model_path,
        output_jsonl=args.output_jsonl,
        output_summary=args.output_summary,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_cases=args.max_cases,
        max_candidate_images=max_candidate_images,
        max_parse_retries=args.max_parse_retries,
        parse_error_policy=args.parse_error_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
