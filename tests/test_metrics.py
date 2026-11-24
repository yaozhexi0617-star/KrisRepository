# -*- coding: utf-8 -*-
"""指标函数的单元测试：python -m pytest tests -q 或 python tests/test_metrics.py"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import ndcg_at_k, recall_at_k, rmse  # noqa: E402


def test_recall_basic():
    assert recall_at_k([1, 2, 3], {2, 5}, 10) == 0.5
    assert recall_at_k([1, 2], {9}, 10) == 0.0
    assert recall_at_k([1], {1}, 0) == 0.0


def test_recall_k_cutoff():
    # 共 11 个候选，13 排在第 11 位：K=10 时不计入，K=11 时命中
    recs = list(range(3, 14))
    assert recall_at_k(recs, {13}, 10) == 0.0
    assert recall_at_k(recs, {13}, 11) == 1.0


def test_ndcg_perfect():
    assert abs(ndcg_at_k([5, 6], {5, 6}, 10) - 1.0) < 1e-9


def test_ndcg_order_matters():
    good = ndcg_at_k([5, 1, 2], {5}, 10)
    bad = ndcg_at_k([1, 2, 5], {5}, 10)
    assert good > bad


def test_ndcg_with_more_relevant_than_k():
    recs = [1, 2, 3]
    rel = {1, 2, 3, 4, 5}
    v = ndcg_at_k(recs, rel, 3)
    assert 0 < v <= 1.0 + 1e-9


def test_rmse():
    assert abs(rmse([3, 3], [3, 3])) < 1e-12
    assert abs(rmse([1, 5], [3, 3]) - 2.0) < 1e-9
    assert math.isnan(rmse([], []))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
