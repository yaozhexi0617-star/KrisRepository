# -*- coding: utf-8 -*-
"""MovieLens（latest-small / 100k）的读取、切分与特征构建。

统一把用户和条目重编号为 0..n-1 的连续下标，后续模型只处理下标。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ML_SUBDIR, RELEVANT_THRESHOLD

GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "(no genres listed)",
]


def load_ratings(root: Path = ML_SUBDIR) -> pd.DataFrame:
    if (root / "ratings.csv").exists():          # latest-small 格式
        df = pd.read_csv(root / "ratings.csv")
        df = df.rename(columns={"userId": "user", "movieId": "item", "timestamp": "ts"})
    else:                                        # 100k 格式
        df = pd.read_csv(root / "u.data", sep="\t",
                         names=["user", "item", "rating", "ts"])
    return df[["user", "item", "rating", "ts"]].astype(
        {"user": np.int32, "item": np.int32, "rating": np.float32, "ts": np.int64})


def load_items(root: Path = ML_SUBDIR) -> pd.DataFrame:
    if (root / "movies.csv").exists():           # latest-small
        df = pd.read_csv(root / "movies.csv")
        df = df.rename(columns={"movieId": "item"})
        df["genres"] = df["genres"].fillna("").apply(
            lambda s: [g for g in s.split("|") if g])
    else:                                        # 100k
        rows = []
        with open(root / "u.item", encoding="latin-1") as f:
            for line in f:
                p = line.rstrip("\n").split("|")
                flags = [int(x) for x in p[5:24]]
                rows.append({"item": int(p[0]), "title": p[1],
                             "genres": [g for g, fl in zip(GENRES[:19], flags) if fl]})
        df = pd.DataFrame(rows)
    df["year"] = df["title"].apply(_parse_year)
    df["item"] = df["item"].astype(np.int32)
    return df


def _parse_year(title: str) -> int:
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(m.group(1)) if m else 0


def _genre_index(g: str) -> int:
    return GENRES.index(g) if g in GENRES else -1


def _genre_matrix(items: pd.DataFrame) -> np.ndarray:
    mat = np.zeros((len(items), len(GENRES)), dtype=np.float32)
    for row, gs in enumerate(items["genres"].values):
        for g in gs:
            j = _genre_index(g)
            if j >= 0:
                mat[row, j] = 1.0
    return mat


def build(min_ratings: int = 5, threshold: float = RELEVANT_THRESHOLD,
          root: Path = ML_SUBDIR) -> dict:
    """返回一个包含训练/测试划分和各类索引的字典。"""
    ratings = load_ratings(root)
    items_all = load_items(root)

    counts = ratings.groupby("user").size()
    keep_users = counts[counts >= min_ratings].index
    ratings = ratings[ratings["user"].isin(keep_users)]
    keep_items = ratings["item"].unique()
    items_all = items_all[items_all["item"].isin(keep_items)].reset_index(drop=True)

    users = np.sort(ratings["user"].unique())
    items = np.sort(ratings["item"].unique())
    u_index = {int(u): k for k, u in enumerate(users)}
    i_index = {int(i): k for k, i in enumerate(items)}

    ratings = ratings.sort_values(["user", "ts"], kind="mergesort").reset_index(drop=True)
    ratings["u"] = ratings["user"].map(u_index).astype(np.int32)
    ratings["i"] = ratings["item"].map(i_index).astype(np.int32)

    last = ratings.groupby("u").tail(1).index
    train = ratings.drop(index=last).reset_index(drop=True)
    test = ratings.loc[last].reset_index(drop=True)

    items_idx = items_all.set_index("item").loc[items].reset_index()
    genre = _genre_matrix(items_idx)
    year = items_idx["year"].to_numpy()
    item_genres = [set(g) for g in items_idx["genres"].values]
    votes = popularity(train, len(items))

    return dict(
        users=users, items=items, u_index=u_index, i_index=i_index,
        train=train, test=test, items_df=items_idx,
        genre=genre, year=year, item_genres=item_genres, votes=votes,
        title=items_idx["title"].to_numpy(),
        n_users=len(users), n_items=len(items),
        threshold=threshold,
    )


def seen_by_user(train: pd.DataFrame, n_users: int) -> list[set]:
    seen = [set() for _ in range(n_users)]
    for u, i in zip(train["u"].values, train["i"].values):
        seen[int(u)].add(int(i))
    return seen


def relevant_by_user(test: pd.DataFrame, n_users: int, threshold: float) -> list[set]:
    rel = [set() for _ in range(n_users)]
    for u, i, r in zip(test["u"].values, test["i"].values, test["rating"].values):
        if r >= threshold:
            rel[int(u)].add(int(i))
    return rel


def popularity(train: pd.DataFrame, n_items: int) -> np.ndarray:
    cnt = np.zeros(n_items, dtype=np.float32)
    for i in train["i"].values:
        cnt[i] += 1
    return cnt


def rating_matrix(train: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    m = np.zeros((n_users, n_items), dtype=np.float32)
    m[train["u"].values, train["i"].values] = train["rating"].values
    return m
