# -*- coding: utf-8 -*-
"""带偏置的矩阵分解（SGD 训练）。

r_hat(u,i) = mu + b_u + b_i + p_u · q_i
用 numpy 实现，避免为一个小模型引入深度学习框架。
"""
from __future__ import annotations

import numpy as np


class MF:
    def __init__(self, n_users: int, n_items: int, factors: int = 32,
                 lr: float = 0.01, reg: float = 0.05, seed: int = 2026):
        rng = np.random.default_rng(seed)
        self.mu = 0.0
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.P = rng.normal(0, 0.1, size=(n_users, factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, size=(n_items, factors)).astype(np.float32)
        self.lr = lr
        self.reg = reg
        self.history: list[float] = []

    # ------------------------------------------------------------------
    def fit(self, u: np.ndarray, i: np.ndarray, r: np.ndarray,
            epochs: int = 30, batch_shuffle: bool = True, seed: int = 2026):
        rng = np.random.default_rng(seed)
        self.mu = float(r.mean())
        n = len(r)
        idx = np.arange(n)
        for ep in range(epochs):
            if batch_shuffle:
                rng.shuffle(idx)
            loss = 0.0
            for t in idx:
                uu, ii, rr = int(u[t]), int(i[t]), float(r[t])
                pu, qi = self.P[uu], self.Q[ii]
                pred = self.mu + self.bu[uu] + self.bi[ii] + float(pu @ qi)
                err = rr - pred
                loss += err * err
                # SGD 更新（先算梯度再更新，避免覆盖）
                self.bu[uu] += self.lr * (err - self.reg * self.bu[uu])
                self.bi[ii] += self.lr * (err - self.reg * self.bi[ii])
                self.P[uu] = pu + self.lr * (err * qi - self.reg * pu)
                self.Q[ii] = qi + self.lr * (err * pu - self.reg * qi)
            rmse = (loss / n) ** 0.5
            self.history.append(rmse)
            print(f"  epoch {ep + 1:02d}/{epochs}  train_rmse={rmse:.4f}")
        return self

    # ------------------------------------------------------------------
    def predict(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        return (self.mu + self.bu[u] + self.bi[i]
                + np.sum(self.P[u] * self.Q[i], axis=1))

    def scores_for(self, u: int) -> np.ndarray:
        """返回该用户对所有条目的预测分（向量化）。"""
        return self.mu + self.bu[u] + self.bi + self.Q @ self.P[u]

    def topk(self, u: int, seen: set[int], k: int = 10,
             mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        s = self.scores_for(u)
        order = np.argsort(-s)
        out = []
        for i in order:
            ii = int(i)
            if ii in seen:
                continue
            if mask is not None and not mask[ii]:
                continue
            out.append((ii, float(s[ii])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------
    def save(self, path: str):
        np.savez(path, mu=self.mu, bu=self.bu, bi=self.bi, P=self.P, Q=self.Q)

    @classmethod
    def load(cls, path: str) -> "MF":
        d = np.load(path)
        m = cls(d["P"].shape[0], d["Q"].shape[0], factors=d["P"].shape[1])
        m.mu = float(d["mu"])
        m.bu, m.bi, m.P, m.Q = d["bu"], d["bi"], d["P"], d["Q"]
        return m
