# -*- coding: utf-8 -*-
"""排序与评分指标：Recall@K、NDCG@K、RMSE、覆盖率。"""
from __future__ import annotations

import math


def recall_at_k(recs: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for i in recs[:k] if i in relevant)
    return hit / len(relevant)


def ndcg_at_k(recs: list[int], relevant: set[int], k: int = 10) -> float:
    """二值相关性下的 NDCG@K。"""
    dcg = 0.0
    for pos, item in enumerate(recs[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(pos + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(p + 1) for p in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def rmse(y_true, y_pred) -> float:
    n = len(y_true)
    if n == 0:
        return float("nan")
    s = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    return math.sqrt(s / n)


def catalog_coverage(all_recs: list[list[int]], n_items: int) -> float:
    seen = set()
    for r in all_recs:
        seen.update(r)
    return len(seen) / n_items if n_items else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
