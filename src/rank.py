# -*- coding: utf-8 -*-
"""硬约束过滤 + 群体重排（max-min 公平）。"""
from __future__ import annotations

import numpy as np

from .constraints import Constraints


def build_mask(c: Constraints, item_genres: list[set[str]], year: np.ndarray,
               votes: np.ndarray, titles: np.ndarray,
               ban: set[int] | None = None) -> np.ndarray:
    """返回布尔掩码：True 表示该条目满足全部硬约束。"""
    n = len(item_genres)
    mask = np.ones(n, dtype=bool)
    for k in range(n):
        gs = item_genres[k]
        if c.include_genres and not (gs & c.include_genres):
            mask[k] = False
            continue
        if c.exclude_genres and (gs & c.exclude_genres):
            mask[k] = False
            continue
        if not (c.year_min <= int(year[k]) <= c.year_max):
            mask[k] = False
            continue
        if votes[k] < c.min_votes:
            mask[k] = False
            continue
        if c.exclude_titles and any(name and name in str(titles[k])
                                    for name in c.exclude_titles):
            mask[k] = False
            continue
        if ban and k in ban:
            mask[k] = False
    return mask


def utility_matrix(scores: np.ndarray) -> np.ndarray:
    """把 (n_users, n_items) 的打分矩阵转成分位效用：1.0 表示该用户榜单第一。

    这样不同模型（MF 的预测分、ItemKNN 的相似度和）都能用同一套满意度指标比较。
    """
    n_items = scores.shape[1]
    order = np.argsort(-scores, axis=1)                 # 每行从高到低的位置
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(n_items)[None, :]
    return 1.0 - ranks / max(1, n_items - 1)


def pool_utility(sub_scores: np.ndarray) -> np.ndarray:
    """池内相对效用：把 (m, P) 的打分矩阵按每位成员在候选池内的排名换算成 0~1。

    与全库分位相比，池内分位不会被“候选池已经筛选过”这件事稀释，能真实反映
    “这部片对这位成员来说是不是前排”。
    """
    m, p = sub_scores.shape
    order = np.argsort(-sub_scores, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(m)[:, None], order] = np.arange(p)[None, :]
    return 1.0 - ranks / max(1, p - 1)


def retrieve_pool(scores: np.ndarray, members: list[int], mask: np.ndarray,
                  n: int = 200) -> np.ndarray:
    """候选池：每位成员在“满足硬约束的条目”里取前 n 名，再取并集。

    注意必须先按 mask 排除不合法条目再取 Top-N，否则会出现
    “池子里全是老片、按年份过滤后为空”的情况。
    """
    pool = np.zeros(scores.shape[1], dtype=bool)
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return pool
    for m in members:
        s = scores[m, valid]
        keep = min(n, len(valid))
        top = valid[np.argpartition(-s, keep - 1)[:keep]]
        pool[top] = True
    return pool


def group_scores(score_mat: np.ndarray, members: list[int],
                 fairness: str = "min") -> np.ndarray:
    """score_mat: (n_users, n_items)。返回 (n_items,) 的群体得分。

    fairness:
      mean  —— 只按平均分排（多数人满意，可能牺牲少数人）
      min   —— max-min：按最不满意的那位成员排
      blend —— 0.5*mean + 0.5*min
    """
    sub = score_mat[members]
    if fairness == "mean":
        return sub.mean(axis=0)
    if fairness == "min":
        return sub.min(axis=0)
    if fairness == "blend":
        return 0.5 * sub.mean(axis=0) + 0.5 * sub.min(axis=0)
    raise ValueError(f"unknown fairness: {fairness}")


def rank(scores: np.ndarray, mask: np.ndarray, seen: set[int] | None = None,
         k: int = 10) -> list[int]:
    """按分数取前 k 个合法且未看过的条目。"""
    s = scores.copy()
    s[~mask] = -1e9
    if seen:
        s[list(seen)] = -1e9
    order = np.argsort(-s)
    out = []
    for i in order:
        if s[i] <= -1e8:
            break
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def satisfaction(scores: np.ndarray, members: list[int], item: int,
                 lo: float = 1.0, hi: float = 5.0) -> list[float]:
    """把预测分归一化到 [0,1]，便于跨用户比较。"""
    return [float(np.clip((scores[m, item] - lo) / (hi - lo), 0, 1)) for m in members]
