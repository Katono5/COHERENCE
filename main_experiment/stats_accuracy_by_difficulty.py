#!/usr/bin/env python3
"""Difficulty-level table with exact accuracy and Kendall tau side by side."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from typing import Dict, List, Optional, Set, Tuple

try:
    from accuracy_table_common import (
        Agg,
        DIFFICULTY_LABELS,
        DIFFICULTY_ORDER,
        difficulty_from_answer,
        draw_table_png,
        format_kendall,
        format_pct,
        infer_model_subset,
        iter_result_jsonl_files,
        load_benchmark_subset_data_ids,
        load_jsonl_records,
        merge_exact_kendall_rows,
        maybe_write_tsv,
        paired_metric_headers,
        parse_exact_correct,
        parse_kendall_tau,
        print_table,
        should_include_record,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from accuracy_table_common import (
        Agg,
        DIFFICULTY_LABELS,
        DIFFICULTY_ORDER,
        difficulty_from_answer,
        draw_table_png,
        format_kendall,
        format_pct,
        infer_model_subset,
        iter_result_jsonl_files,
        load_benchmark_subset_data_ids,
        load_jsonl_records,
        merge_exact_kendall_rows,
        maybe_write_tsv,
        paired_metric_headers,
        parse_exact_correct,
        parse_kendall_tau,
        print_table,
        should_include_record,
    )


def _scan_difficulty_file(
    jsonl_path: Path,
    allowed_data_ids: Optional[Set[str]],
) -> Tuple[str, Dict[str, Agg], Agg]:
    model, _subset = infer_model_subset(jsonl_path)
    bucket_aggs: Dict[str, Agg] = {}
    overall_agg = Agg()

    for record in load_jsonl_records(jsonl_path):
        if not should_include_record(record, allowed_data_ids):
            continue

        difficulty = difficulty_from_answer(record.get("answer"))
        if difficulty not in DIFFICULTY_LABELS:
            continue
        if difficulty not in bucket_aggs:
            bucket_aggs[difficulty] = Agg()

        is_correct = parse_exact_correct(record)
        kendall_tau = parse_kendall_tau(record)
        bucket_aggs[difficulty].add(is_correct, kendall_tau)
        overall_agg.add(is_correct, kendall_tau)

    return model, bucket_aggs, overall_agg


def _aggregate_difficulty_scores(
    results_dir: Path,
    allowed_data_ids: Optional[Set[str]],
    num_workers: int,
) -> Tuple[Dict[str, Dict[str, Agg]], Dict[str, Agg]]:
    model_bucket_agg: Dict[str, Dict[str, Agg]] = {}
    model_overall_agg: Dict[str, Agg] = {}
    jsonl_files = list(iter_result_jsonl_files(results_dir))
    worker_count = max(1, num_workers)

    def merge_scan_result(scan_result: Tuple[str, Dict[str, Agg], Agg]) -> None:
        model, bucket_aggs, overall_agg = scan_result
        if overall_agg.total <= 0:
            return
        if model not in model_bucket_agg:
            model_bucket_agg[model] = {}
        if model not in model_overall_agg:
            model_overall_agg[model] = Agg()

        for bucket, bucket_agg in bucket_aggs.items():
            if bucket not in model_bucket_agg[model]:
                model_bucket_agg[model][bucket] = Agg()
            model_bucket_agg[model][bucket].merge(bucket_agg)
        model_overall_agg[model].merge(overall_agg)

    if worker_count <= 1 or len(jsonl_files) <= 1:
        for jsonl_path in jsonl_files:
            merge_scan_result(_scan_difficulty_file(jsonl_path, allowed_data_ids))
        return model_bucket_agg, model_overall_agg

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for scan_result in executor.map(
            lambda path: _scan_difficulty_file(path, allowed_data_ids),
            jsonl_files,
        ):
            merge_scan_result(scan_result)

    return model_bucket_agg, model_overall_agg


def _build_difficulty_rows(
    model_bucket_agg: Dict[str, Dict[str, Agg]],
    model_overall_agg: Dict[str, Agg],
    metric: str,
    sort_by: str,
    descending: bool,
) -> List[Dict[str, str]]:
    rows_with_score: List[tuple[Dict[str, str], float]] = []
    for model in sorted(model_overall_agg.keys()):
        if model_overall_agg[model].total <= 0:
            continue

        row: Dict[str, str] = {"Model": model}
        for bucket, label in DIFFICULTY_ORDER:
            agg = model_bucket_agg[model].get(bucket, Agg())
            if metric == "kendall":
                row[label] = format_kendall(agg)
            else:
                row[label] = format_pct(agg)

        if metric == "kendall":
            row["Overall"] = format_kendall(model_overall_agg[model])
            score = model_overall_agg[model].kendall_tau_avg
        else:
            row["Overall"] = format_pct(model_overall_agg[model])
            score = model_overall_agg[model].exact_accuracy
        rows_with_score.append((row, score))

    if sort_by == "overall":
        rows_with_score.sort(key=lambda x: x[1], reverse=descending)
    else:
        rows_with_score.sort(key=lambda x: x[0]["Model"], reverse=descending)

    return [row for row, _score in rows_with_score]


def collect_difficulty_rows(
    results_dir: Path,
    metric: str = "exact",
    sort_by: str = "model",
    descending: bool = False,
    allowed_data_ids: Optional[Set[str]] = None,
    num_workers: int = 1,
) -> List[Dict[str, str]]:
    model_bucket_agg, model_overall_agg = _aggregate_difficulty_scores(
        results_dir=results_dir,
        allowed_data_ids=allowed_data_ids,
        num_workers=num_workers,
    )
    return _build_difficulty_rows(
        model_bucket_agg=model_bucket_agg,
        model_overall_agg=model_overall_agg,
        metric=metric,
        sort_by=sort_by,
        descending=descending,
    )


def collect_difficulty_rows_combined(
    results_dir: Path,
    sort_by: str = "model",
    descending: bool = False,
    allowed_data_ids: Optional[Set[str]] = None,
    num_workers: int = 1,
) -> List[Dict[str, str]]:
    model_bucket_agg, model_overall_agg = _aggregate_difficulty_scores(
        results_dir=results_dir,
        allowed_data_ids=allowed_data_ids,
        num_workers=num_workers,
    )
    rows_exact = _build_difficulty_rows(
        model_bucket_agg=model_bucket_agg,
        model_overall_agg=model_overall_agg,
        metric="exact",
        sort_by=sort_by,
        descending=descending,
    )
    rows_kendall = _build_difficulty_rows(
        model_bucket_agg=model_bucket_agg,
        model_overall_agg=model_overall_agg,
        metric="kendall",
        sort_by=sort_by,
        descending=descending,
    )
    category_labels = [label for _key, label in DIFFICULTY_ORDER]
    return merge_exact_kendall_rows(rows_exact, rows_kendall, category_labels)


def _resolve_path_with_repo_fallback(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    repo_candidate = Path(__file__).resolve().parents[1] / path
    if repo_candidate.exists():
        return repo_candidate
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "按题目难度统计并输出单表：每个难度同时包含 "
            "Exact Match Accuracy 和 Kendall Tau（百分制）两列。"
            "难度由 len(answer) 定义：len<7 为 easy，7-12 为 medium，>12 为 hard。"
        )
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="结果目录")
    parser.add_argument(
        "--benchmark_subset_dir",
        type=Path,
        default=Path("datasets/benchmark_data/full_benchmark_7670"),
        help="用于过滤统计范围的 benchmark 子集目录（默认 7670）",
    )
    parser.add_argument(
        "--sort_by",
        choices=["model", "overall"],
        default="model",
        help="排序方式（默认按模型名）",
    )
    parser.add_argument("--descending", action="store_true", help="降序")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 1) * 2)),
        help="并行统计线程数",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("results/difficulty_exact_kendall_table.png"),
        help="Difficulty 合并表图片路径",
    )
    parser.add_argument(
        "--output_tsv",
        type=Path,
        default=None,
        help="可选：Difficulty 合并表 TSV 路径",
    )
    # Backward-compatible aliases from the previous two-table version.
    parser.add_argument("--output_png_exact", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output_png_kendall", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output_tsv_exact", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output_tsv_kendall", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.results_dir = _resolve_path_with_repo_fallback(args.results_dir)
    args.benchmark_subset_dir = _resolve_path_with_repo_fallback(args.benchmark_subset_dir)

    if not args.results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {args.results_dir}")
    subset_data_ids = load_benchmark_subset_data_ids(args.benchmark_subset_dir)
    if not subset_data_ids:
        raise ValueError(f"no data_id found in benchmark subset dir: {args.benchmark_subset_dir}")

    output_png = args.output_png
    legacy_png_values = [p for p in [args.output_png_exact, args.output_png_kendall] if p is not None]
    if len(legacy_png_values) > 1:
        raise ValueError(
            "This script now outputs one combined table. "
            "Please provide at most one of --output_png_exact/--output_png_kendall."
        )
    if legacy_png_values:
        output_png = legacy_png_values[0]

    output_tsv = args.output_tsv
    legacy_tsv_values = [p for p in [args.output_tsv_exact, args.output_tsv_kendall] if p is not None]
    if len(legacy_tsv_values) > 1:
        raise ValueError(
            "This script now outputs one combined table. "
            "Please provide at most one of --output_tsv_exact/--output_tsv_kendall."
        )
    if legacy_tsv_values:
        output_tsv = legacy_tsv_values[0]

    headers = paired_metric_headers([label for _key, label in DIFFICULTY_ORDER])
    rows = collect_difficulty_rows_combined(
        results_dir=args.results_dir,
        sort_by=args.sort_by,
        descending=args.descending,
        allowed_data_ids=subset_data_ids,
        num_workers=max(1, args.num_workers),
    )

    print("Difficulty exact+kendall table\n")
    print(
        f"subset_filter={args.benchmark_subset_dir}, "
        f"subset_size={len(subset_data_ids)}, "
        f"num_workers={max(1, args.num_workers)}"
    )
    print_table(rows, headers)

    draw_table_png(
        rows=rows,
        headers=headers,
        title="Difficulty Exact Match + Kendall Tau (%)",
        output_png=output_png,
    )
    print(f"\nSaved combined table image to: {output_png}")

    if output_tsv is not None:
        maybe_write_tsv(rows, headers, output_tsv)
        print(f"Saved combined TSV to: {output_tsv}")


if __name__ == "__main__":
    main()
