#!/usr/bin/env python3
"""Domain-level table with exact accuracy and Kendall tau side by side."""

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
        DOMAIN_LABELS,
        DOMAIN_ORDER,
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
        DOMAIN_LABELS,
        DOMAIN_ORDER,
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


def _scan_domain_file(
    jsonl_path: Path,
    allowed_data_ids: Optional[Set[str]],
) -> Tuple[str, str, Agg, Agg]:
    model, subset = infer_model_subset(jsonl_path)
    subset_agg = Agg()
    overall_agg = Agg()

    if subset not in DOMAIN_LABELS:
        return model, subset, subset_agg, overall_agg

    for record in load_jsonl_records(jsonl_path):
        if not should_include_record(record, allowed_data_ids):
            continue
        is_correct = parse_exact_correct(record)
        kendall_tau = parse_kendall_tau(record)
        subset_agg.add(is_correct, kendall_tau)
        overall_agg.add(is_correct, kendall_tau)

    return model, subset, subset_agg, overall_agg


def _aggregate_domain_scores(
    results_dir: Path,
    allowed_data_ids: Optional[Set[str]],
    num_workers: int,
) -> Tuple[Dict[str, Dict[str, Agg]], Dict[str, Agg]]:
    model_domain_agg: Dict[str, Dict[str, Agg]] = {}
    model_overall_agg: Dict[str, Agg] = {}
    jsonl_files = list(iter_result_jsonl_files(results_dir))
    worker_count = max(1, num_workers)

    def merge_scan_result(scan_result: Tuple[str, str, Agg, Agg]) -> None:
        model, subset, subset_agg, overall_agg = scan_result
        if overall_agg.total <= 0:
            return
        if model not in model_domain_agg:
            model_domain_agg[model] = {}
        if subset not in model_domain_agg[model]:
            model_domain_agg[model][subset] = Agg()
        if model not in model_overall_agg:
            model_overall_agg[model] = Agg()
        model_domain_agg[model][subset].merge(subset_agg)
        model_overall_agg[model].merge(overall_agg)

    if worker_count <= 1 or len(jsonl_files) <= 1:
        for jsonl_path in jsonl_files:
            merge_scan_result(_scan_domain_file(jsonl_path, allowed_data_ids))
        return model_domain_agg, model_overall_agg

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for scan_result in executor.map(
            lambda path: _scan_domain_file(path, allowed_data_ids),
            jsonl_files,
        ):
            merge_scan_result(scan_result)

    return model_domain_agg, model_overall_agg


def _build_domain_rows(
    model_domain_agg: Dict[str, Dict[str, Agg]],
    model_overall_agg: Dict[str, Agg],
    metric: str,
    sort_by: str,
    descending: bool,
) -> List[Dict[str, str]]:
    rows_with_score: List[tuple[Dict[str, str], float]] = []
    for model in sorted(model_overall_agg.keys()):
        row: Dict[str, str] = {"Model": model}
        for subset, label in DOMAIN_ORDER:
            agg = model_domain_agg[model].get(subset, Agg())
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


def collect_domain_rows(
    results_dir: Path,
    metric: str = "exact",
    sort_by: str = "model",
    descending: bool = False,
    allowed_data_ids: Optional[Set[str]] = None,
    num_workers: int = 1,
) -> List[Dict[str, str]]:
    model_domain_agg, model_overall_agg = _aggregate_domain_scores(
        results_dir=results_dir,
        allowed_data_ids=allowed_data_ids,
        num_workers=num_workers,
    )
    return _build_domain_rows(
        model_domain_agg=model_domain_agg,
        model_overall_agg=model_overall_agg,
        metric=metric,
        sort_by=sort_by,
        descending=descending,
    )


def collect_domain_rows_combined(
    results_dir: Path,
    sort_by: str = "model",
    descending: bool = False,
    allowed_data_ids: Optional[Set[str]] = None,
    num_workers: int = 1,
) -> List[Dict[str, str]]:
    model_domain_agg, model_overall_agg = _aggregate_domain_scores(
        results_dir=results_dir,
        allowed_data_ids=allowed_data_ids,
        num_workers=num_workers,
    )
    rows_exact = _build_domain_rows(
        model_domain_agg=model_domain_agg,
        model_overall_agg=model_overall_agg,
        metric="exact",
        sort_by=sort_by,
        descending=descending,
    )
    rows_kendall = _build_domain_rows(
        model_domain_agg=model_domain_agg,
        model_overall_agg=model_overall_agg,
        metric="kendall",
        sort_by=sort_by,
        descending=descending,
    )
    category_labels = [label for _key, label in DOMAIN_ORDER]
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
            "Generate a single domain-level table: each domain includes "
            "Exact Match Accuracy and Kendall Tau (percentage), plus Overall columns."
        )
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Results directory")
    parser.add_argument(
        "--benchmark_subset_dir",
        type=Path,
        default=Path("datasets/benchmark_data"),
        help="Benchmark subset directory used to filter the statistics scope (default: datasets/benchmark_data)",
    )
    parser.add_argument(
        "--sort_by",
        choices=["model", "overall"],
        default="model",
        help="Sort mode (default: model name)",
    )
    parser.add_argument("--descending", action="store_true", help="Sort in descending order")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(16, min(32, (os.cpu_count() or 1) * 2)),
        help="Number of worker threads for parallel aggregation",
    )
    parser.add_argument(
        "--output_png",
        type=Path,
        default=Path("results/domain_exact_kendall_table.png"),
        help="Output path for merged domain table image",
    )
    parser.add_argument(
        "--output_tsv",
        type=Path,
        default=None,
        help="Optional: output path for merged domain table TSV",
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

    headers = paired_metric_headers([label for _key, label in DOMAIN_ORDER])
    rows = collect_domain_rows_combined(
        results_dir=args.results_dir,
        sort_by=args.sort_by,
        descending=args.descending,
        allowed_data_ids=subset_data_ids,
        num_workers=max(1, args.num_workers),
    )

    print("Domain exact+kendall table\n")
    print(
        f"subset_filter={args.benchmark_subset_dir}, "
        f"subset_size={len(subset_data_ids)}, "
        f"num_workers={max(1, args.num_workers)}"
    )
    print_table(rows, headers)

    draw_table_png(
        rows=rows,
        headers=headers,
        title="Domain Exact Match + Kendall Tau (%)",
        output_png=output_png,
    )
    print(f"\nSaved combined table image to: {output_png}")

    if output_tsv is not None:
        maybe_write_tsv(rows, headers, output_tsv)
        print(f"Saved combined TSV to: {output_tsv}")


if __name__ == "__main__":
    main()
