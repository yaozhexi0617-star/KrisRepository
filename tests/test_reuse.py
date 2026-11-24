# -*- coding: utf-8 -*-
"""组件复用核验：固定随机种子后 KMeans 结果是否可复现。

    python tests/test_reuse.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D     # noqa: E402
from src.config import SEED   # noqa: E402


def user_profiles(b):
    prof = np.zeros((b["n_users"], b["genre"].shape[1]), dtype=np.float32)
    cnt = np.zeros(b["n_users"], dtype=np.float32)
    tr = b["train"]
    for u, i, r in zip(tr["u"].values, tr["i"].values, tr["rating"].values):
        prof[u] += b["genre"][i] * (r / 5.0)
        cnt[u] += 1
    cnt[cnt == 0] = 1
    return prof / cnt[:, None]


def fit_once(X, seed):
    return KMeans(n_clusters=8, n_init=10, random_state=seed).fit(X).cluster_centers_


def main():
    b = D.build()
    X = user_profiles(b)

    c1 = fit_once(X, SEED)
    c2 = fit_once(X, SEED)
    d_same = float(np.abs(c1 - c2).max())

    c3 = fit_once(X, SEED + 1)
    # 不同种子：簇编号可能不同，按最近中心配对后比较
    d_cross = max(float(np.abs(c1[i] - c3[np.argmin(((c3 - c1[i]) ** 2).sum(axis=1))]).max())
                  for i in range(len(c1)))

    print(f"同一 random_state 两次运行，中心最大差异 = {d_same:.2e}")
    print(f"不同 random_state 配对后，中心最大差异 = {d_cross:.4f}")
    assert d_same < 1e-6, "同种子不可复现"
    print("OK: 固定 random_state 结果可复现")


if __name__ == "__main__":
    main()
