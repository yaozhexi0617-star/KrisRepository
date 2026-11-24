# -*- coding: utf-8 -*-
"""两个基线：全局热门榜与 ItemKNN（基于余弦相似度）。"""
from __future__ import annotations

import numpy as np


class Popular:
    """按训练集评分人数排序，是最常用也最难打败的基线。"""

    def __init__(self, train, n_items: int):
        self.score = np.zeros(n_items, dtype=np.float32)
        for i in train["i"].values:
            self.score[i] += 1.0

    def topk(self, seen: set[int], k: int = 10) -> list[int]:
        order = np.argsort(-self.score)
        out = []
        for i in order:
            if int(i) in seen:
                continue
            out.append(int(i))
            if len(out) >= k:
                break
        return out

    def predict(self, u: int, i: int) -> float:      # 用于 RMSE 时退化为均值
        return float(self.score[i])


class ItemKNN:
    """item-item 余弦相似度；把用户已评分项按相似度加权求和作为打分。"""

    def __init__(self, rating_mat: np.ndarray, shrink: float = 100.0):
        # 列归一化后点积 = 余弦相似度
        norms = np.linalg.norm(rating_mat, axis=0)
        norms[norms == 0] = 1e-9
        X = rating_mat / norms
        sim = X.T @ X
        np.fill_diagonal(sim, 0.0)
        # shrink：惩罚评分人数少的条目，避免冷门项相似度虚高
        cnt = (rating_mat > 0).sum(axis=0)
        sim *= cnt[None, :] / (cnt[None, :] + shrink)
        self.sim = sim.astype(np.float32)
        self.R = rating_mat

    def scores_for(self, u: int) -> np.ndarray:
        rated = self.R[u]
        return rated @ self.sim

    def topk(self, u: int, seen: set[int], k: int = 10) -> list[int]:
        s = self.scores_for(u)
        order = np.argsort(-s)
        out = []
        for i in order:
            if int(i) in seen or s[i] <= 0:
                continue
            out.append(int(i))
            if len(out) >= k:
                break
        return out
