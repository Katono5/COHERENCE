#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared metric helpers for arrangement evaluation."""

from typing import Any, Dict, List, Optional


def _resolve_limit(answer_len: int, num_placeholders: Optional[int]) -> int:
    if answer_len <= 0:
        return 0
    if num_placeholders is None:
        return answer_len
    try:
        limit = int(num_placeholders)
    except (TypeError, ValueError):
        return answer_len
    if limit <= 0:
        return 0
    return min(limit, answer_len)


def kendall_tau_mapped_0_1(
    pred: Any,
    answer: Any,
    num_placeholders: Optional[int] = None,
) -> float:
    """Compute mapped Kendall tau in [0, 1] from two index lists.

    Mapping rule:
    - tau = (C - D) / (C + D) in [-1, 1]
    - mapped = (tau + 1) / 2 = C / (C + D) in [0, 1]

    The implementation is robust to malformed outputs:
    - Non-list inputs -> 0.0
    - Fewer than two comparable items -> 0.0
    - Duplicates -> keep first occurrence
    - Missing items -> compare only intersection
    """
    if not isinstance(pred, list) or not isinstance(answer, list):
        return 0.0

    n = _resolve_limit(len(answer), num_placeholders)
    if n < 2:
        return 0.0

    answer_slice: List[Any] = answer[:n]
    pred_slice: List[Any] = pred[:n]

    pos_answer: Dict[Any, int] = {}
    for idx, val in enumerate(answer_slice):
        if val not in pos_answer:
            pos_answer[val] = idx

    pos_pred: Dict[Any, int] = {}
    for idx, val in enumerate(pred_slice):
        if val not in pos_pred:
            pos_pred[val] = idx

    common = [val for val in pos_answer if val in pos_pred]
    m = len(common)
    if m < 2:
        return 0.0

    concordant = 0
    discordant = 0
    for i in range(m):
        a = common[i]
        for j in range(i + 1, m):
            b = common[j]
            da = pos_answer[a] - pos_answer[b]
            dp = pos_pred[a] - pos_pred[b]
            sign = da * dp
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1

    denom = concordant + discordant
    if denom <= 0:
        return 0.0

    return concordant / denom
