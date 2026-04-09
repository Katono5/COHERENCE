#!/usr/bin/env python3
"""Shared helpers for accuracy table scripts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

try:
    from metrics import kendall_tau_mapped_0_1
except ImportError:
    from main_experiment.metrics import kendall_tau_mapped_0_1


DOMAIN_LABELS: Dict[str, str] = {
    "cooking": "Cooking",
    "science": "Science",
    "storybird": "Storybird",
    "wikihow": "WikiHow",
}
DOMAIN_ORDER: List[Tuple[str, str]] = [
    ("cooking", DOMAIN_LABELS["cooking"]),
    ("science", DOMAIN_LABELS["science"]),
    ("storybird", DOMAIN_LABELS["storybird"]),
    ("wikihow", DOMAIN_LABELS["wikihow"]),
]

DIFFICULTY_LABELS: Dict[str, str] = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}
DIFFICULTY_ORDER: List[Tuple[str, str]] = [
    ("easy", DIFFICULTY_LABELS["easy"]),
    ("medium", DIFFICULTY_LABELS["medium"]),
    ("hard", DIFFICULTY_LABELS["hard"]),
]

_RESULT_NAME_RE = re.compile(r"^(?P<subset>.+?)_(?:vllm|api)_eval\.jsonl$")


@dataclass
class Agg:
    total: int = 0
    exact_correct: int = 0
    kendall_tau_sum: float = 0.0

    def add(self, is_correct: bool, kendall_tau: float) -> None:
        self.total += 1
        self.exact_correct += 1 if bool(is_correct) else 0
        self.kendall_tau_sum += float(kendall_tau)

    def merge(self, other: "Agg") -> None:
        self.total += other.total
        self.exact_correct += other.exact_correct
        self.kendall_tau_sum += other.kendall_tau_sum

    @property
    def exact_accuracy(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.exact_correct / self.total

    @property
    def kendall_tau_avg(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.kendall_tau_sum / self.total


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value != value:  # nan
            return None
        return float(value) != 0.0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # nan
        return None
    return parsed


def load_jsonl_records(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def iter_result_jsonl_files(results_dir: Path) -> Iterator[Path]:
    for path in sorted(results_dir.rglob("*_eval.jsonl")):
        if path.is_file():
            yield path


def infer_model_subset(jsonl_path: Path) -> Tuple[str, str]:
    model = jsonl_path.parent.name
    match = _RESULT_NAME_RE.match(jsonl_path.name)
    if match is not None:
        subset = match.group("subset")
    else:
        subset = jsonl_path.stem.split("_", 1)[0]
    return model, subset


def load_benchmark_subset_data_ids(benchmark_subset_dir: Path) -> Set[str]:
    files: List[Path] = []
    if benchmark_subset_dir.is_file():
        files = [benchmark_subset_dir]
    elif benchmark_subset_dir.is_dir():
        files = sorted(benchmark_subset_dir.glob("*.jsonl"))
    else:
        return set()

    data_ids: Set[str] = set()
    for jsonl_path in files:
        for record in load_jsonl_records(jsonl_path):
            data_id = str(record.get("data_id", "")).strip()
            if data_id:
                data_ids.add(data_id)
    return data_ids


def should_include_record(record: Dict[str, Any], allowed_data_ids: Optional[Set[str]]) -> bool:
    if allowed_data_ids is None:
        return True
    data_id = str(record.get("data_id", "")).strip()
    return data_id in allowed_data_ids


def parse_exact_correct(record: Dict[str, Any]) -> bool:
    parsed = _parse_bool(record.get("exact_correct"))
    if parsed is not None:
        return parsed

    pred = record.get("prediction")
    answer = record.get("answer")
    if isinstance(pred, list) and isinstance(answer, list):
        return pred == answer
    return False


def parse_kendall_tau(record: Dict[str, Any]) -> float:
    partial_metric = str(record.get("partial_metric", "")).strip().lower()
    if partial_metric == "kendall_tau_0_1":
        parsed = _to_float_or_none(record.get("partial_score"))
        if parsed is not None:
            return parsed

    pred = record.get("prediction")
    answer = record.get("answer")
    if isinstance(pred, list) and isinstance(answer, list):
        return float(kendall_tau_mapped_0_1(pred, answer))
    return 0.0


def difficulty_from_answer(answer: Any) -> str:
    if not isinstance(answer, list):
        return "unknown"
    n = len(answer)
    if n < 7:
        return "easy"
    if n <= 12:
        return "medium"
    return "hard"


def format_pct(agg: Agg) -> str:
    if agg.total <= 0:
        return "-"
    return f"{agg.exact_accuracy * 100:.2f}%"


def format_kendall(agg: Agg) -> str:
    if agg.total <= 0:
        return "-"
    return f"{agg.kendall_tau_avg * 100:.2f}%"


def paired_metric_headers(category_labels: Sequence[str]) -> List[str]:
    headers = ["Model"]
    for label in category_labels:
        headers.append(f"{label} Exact")
        headers.append(f"{label} Kendall")
    headers.append("Overall Exact")
    headers.append("Overall Kendall")
    return headers


def merge_exact_kendall_rows(
    rows_exact: Sequence[Dict[str, str]],
    rows_kendall: Sequence[Dict[str, str]],
    category_labels: Sequence[str],
) -> List[Dict[str, str]]:
    exact_by_model = {row.get("Model", ""): row for row in rows_exact}
    kendall_by_model = {row.get("Model", ""): row for row in rows_kendall}

    ordered_models: List[str] = []
    for row in rows_exact:
        model = row.get("Model", "")
        if model and model not in ordered_models:
            ordered_models.append(model)
    for row in rows_kendall:
        model = row.get("Model", "")
        if model and model not in ordered_models:
            ordered_models.append(model)

    merged_rows: List[Dict[str, str]] = []
    for model in ordered_models:
        exact_row = exact_by_model.get(model, {})
        kendall_row = kendall_by_model.get(model, {})
        row: Dict[str, str] = {"Model": model}
        for label in category_labels:
            row[f"{label} Exact"] = str(exact_row.get(label, "-"))
            row[f"{label} Kendall"] = str(kendall_row.get(label, "-"))
        row["Overall Exact"] = str(exact_row.get("Overall", "-"))
        row["Overall Kendall"] = str(kendall_row.get("Overall", "-"))
        merged_rows.append(row)

    return merged_rows


def print_table(rows: Sequence[Dict[str, str]], headers: Sequence[str]) -> None:
    widths: Dict[str, int] = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))


def maybe_write_tsv(rows: Sequence[Dict[str, str]], headers: Sequence[str], output_tsv: Path) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(headers) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")


def draw_table_png(
    rows: Sequence[Dict[str, str]],
    headers: Sequence[str],
    title: str,
    output_png: Path,
) -> None:
    # Keep a deterministic, dependency-light rendering path via Pillow.
    widths: Dict[str, int] = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)
    lines: List[str] = [title, "", header_line, separator]
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))

    font = ImageFont.load_default()
    line_height = 14
    try:
        bbox = font.getbbox("M")
        char_width = max(6, bbox[2] - bbox[0])
        line_height = max(14, bbox[3] - bbox[1] + 4)
    except Exception:
        char_width = 7

    max_chars = max((len(line) for line in lines), default=1)
    img_w = max(800, max_chars * char_width + 20)
    img_h = max(120, len(lines) * line_height + 20)

    image = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    output_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_png)

