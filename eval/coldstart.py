# -*- coding: utf-8 -*-
"""冷启动实验：只用前 5 条评分，靠“用户分群先验”预测其余评分。

新用户没有任何历史时，协同过滤无法工作。这里的做法是：
  1) 用全体用户的类型偏好向量做 KMeans 分群；
  2) 新用户凭 5 条评分估计自己的类型画像 → 归到最近的簇；
  3) 用簇内条目的平均分作为先验（按评分人数收缩），与全局均值、用户均值对照。

    python eval/coldstart.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as D          # noqa: E402
from src.config import SEED        # noqa: E402
from src.metrics import rmse       # noqa: E402

N_ONBOARD = 5
N_CLUSTERS = 8


def main():
    b = D.build()
    r = D.load_ratings()
    items = b["items"]
    i_index = b["i_index"]
    n_users, n_items = b["n_users"], b["n_items"]
    genre = b["genre"]           # (n_items, 19)

    # 全体用户的类型画像（用全部评分，仅用于聚类，不参与预测）
    prof_all = np.zeros((n_users, genre.shape[1]), dtype=np.float32)
    cnt = np.zeros(n_users, dtype=np.float32)
    drop = r[r["item"].isin(i_index)]
    for u, i, v in zip(drop["user"].values, drop["item"].values, drop["rating"].values):
        if int(u) not in b["u_index"]:
            continue
        prof_all[b["u_index"][int(u)]] += genre[i_index[int(i)]] * (v / 5.0)
        cnt[b["u_index"][int(u)]] += 1
    cnt[cnt == 0] = 1
    prof_all /= cnt[:, None]

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(prof_all)
    labels = km.labels_

    # 簇内条目平均分（排除目标用户自身）
    item_sum = np.zeros((N_CLUSTERS, n_items), dtype=np.float64)
    item_cnt = np.zeros((N_CLUSTERS, n_items), dtype=np.float64)
    for u in range(n_users):
        rows = drop[drop["user"] == b["users"][u]]
        for i, v in zip(rows["item"].values, rows["rating"].values):
            item_sum[labels[u], i_index[int(i)]] += v
            item_cnt[labels[u], i_index[int(i)]] += 1
    cluster_mean = np.divide(item_sum, np.maximum(item_cnt, 1))

    gi = np.zeros(n_items, dtype=np.float64)
    gcnt = np.zeros(n_items, dtype=np.float64)
    for u in range(n_users):
        rows = drop[drop["user"] == b["users"][u]]
        for i, v in zip(rows["item"].values, rows["rating"].values):
            gi[i_index[int(i)]] += v
            gcnt[i_index[int(i)]] += 1
    global_item_mean = np.divide(gi, np.maximum(gcnt, 1))
    mu = float(drop["rating"].mean())

    err = {"global": [], "user_mean": [], "cluster": []}
    for u in range(n_users):
        rows = drop[drop["user"] == b["users"][u]].sort_values("ts")
        if len(rows) <= N_ONBOARD + 2:
            continue
        head, tail = rows.iloc[:N_ONBOARD], rows.iloc[N_ONBOARD:]
        u_mean = float(head["rating"].mean())

        # 用 5 条评分估计画像 → 归簇
        p = np.zeros(genre.shape[1], dtype=np.float32)
        for i, v in zip(head["item"].values, head["rating"].values):
            p += genre[i_index[int(i)]] * (v / 5.0)
        if np.linalg.norm(p) == 0:
            continue
        p /= np.linalg.norm(p)
        centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        c = int(np.argmax(centers @ p))

        for i, v in zip(tail["item"].values, tail["rating"].values):
            k = i_index[int(i)]
            # 收缩：簇内样本少时退回全局条目均值，再退回全局均值
            n_c = item_cnt[c, k]
            prior = (cluster_mean[c, k] * n_c + global_item_mean[k] * 20) / (n_c + 20)
            pred_c = 0.5 * prior + 0.5 * u_mean
            err["global"].append(abs(v - mu))
            err["user_mean"].append(abs(v - u_mean))
            err["cluster"].append(abs(v - pred_c))

    out = {}
    for k, v in err.items():
        v = np.array(v)
        out[k] = dict(rmse=float(np.sqrt((v ** 2).mean())), mae=float(v.mean()), n=int(len(v)))
        print(f"{k:<10} RMSE={out[k]['rmse']:.4f}  MAE={out[k]['mae']:.4f}  n={out[k]['n']}")

    path = ROOT / "results" / "coldstart.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
