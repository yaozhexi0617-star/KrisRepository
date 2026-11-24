# -*- coding: utf-8 -*-
"""一键复现全部实验：排序指标、评分误差、群体重排对照与消融。

    python eval/run_all.py                # 全流程（会训练 MF，约 1~3 分钟）
    python eval/run_all.py --quick        # 少训练几轮，快速冒烟
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as D                        # noqa: E402
from src.baselines import ItemKNN, Popular       # noqa: E402
from src.config import (K, MF_EPOCHS, MF_FACTORS, MF_LR, MF_REG,  # noqa: E402
                        RESULT_DIR, SEED)
from src.constraints import Constraints          # noqa: E402
from src.metrics import (catalog_coverage, mean, ndcg_at_k,      # noqa: E402
                         recall_at_k, rmse)
from src.mf import MF                            # noqa: E402
from src.rank import (build_mask, group_scores, pool_utility, rank,  # noqa: E402
                      retrieve_pool, utility_matrix)

DEMO_CONSTRAINTS = [
    ("不要恐怖片", Constraints(n_people=3, exclude_genres={"Horror"})),
    ("轻松的喜剧", Constraints(n_people=3, include_genres={"Comedy"})),
    ("2010 年以后的", Constraints(n_people=3, year_min=2010)),
    ("热门的动画片", Constraints(n_people=4, include_genres={"Animation"}, min_votes=50)),
]


def user_profiles(b):
    """用户类型偏好画像：按评分加权的类型均值向量。"""
    prof = np.zeros((b["n_users"], b["genre"].shape[1]), dtype=np.float32)
    cnt = np.zeros(b["n_users"], dtype=np.float32)
    tr = b["train"]
    for u, i, r in zip(tr["u"].values, tr["i"].values, tr["rating"].values):
        prof[u] += b["genre"][i] * (r / 5.0)
        cnt[u] += 1
    cnt[cnt == 0] = 1
    return prof / cnt[:, None]


def cosine_matrix(P):
    norm = np.linalg.norm(P, axis=1)
    norm[norm == 0] = 1e-9
    X = P / norm[:, None]
    return X @ X.T


def eval_ranking(name, recommend_fn, b, rel, seen, n_items, allow=None):
    recs, rr, nn = [], [], []
    for u in range(b["n_users"]):
        if not rel[u]:
            continue
        r = recommend_fn(u, seen[u])
        if allow is not None:
            r = [i for i in r if allow[i]][:K]
        recs.append(r)
        rr.append(recall_at_k(r, rel[u], K))
        nn.append(ndcg_at_k(r, rel[u], K))
    return dict(model=name, recall=mean(rr), ndcg=mean(nn),
                coverage=catalog_coverage(recs, n_items), n_users=len(rr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    epochs = 3 if args.quick else MF_EPOCHS

    t0 = time.time()
    print("[1/5] 读数据与切分 ...")
    b = D.build()
    print(f"      users={b['n_users']} items={b['n_items']} "
          f"train={len(b['train'])} test={len(b['test'])}")

    rel = D.relevant_by_user(b["test"], b["n_users"], b["threshold"])
    seen = D.seen_by_user(b["train"], b["n_users"])
    n_items = b["n_items"]

    print("[2/5] 基线：热门榜 / ItemKNN ...")
    pop = Popular(b["train"], n_items)
    R = D.rating_matrix(b["train"], b["n_users"], n_items)
    knn = ItemKNN(R)
    # 统一协议：只推荐训练集中评分人数 >= 20 的条目，避免冷门条目靠噪声上位
    allow = b["votes"] >= 20
    pool_n = 60

    rows = []
    rows.append(eval_ranking("popular", lambda u, s: pop.topk(s, pool_n),
                             b, rel, seen, n_items, allow))
    rows.append(eval_ranking("itemknn", lambda u, s: knn.topk(u, s, pool_n),
                             b, rel, seen, n_items, allow))
    for r in rows:
        print(f"      {r['model']:<10} recall@{K}={r['recall']:.4f} "
              f"ndcg@{K}={r['ndcg']:.4f} coverage={r['coverage']:.3f}")

    print(f"[3/5] 训练矩阵分解（{epochs} epochs, {MF_FACTORS} 维）...")
    tr = b["train"]
    mf = MF(b["n_users"], n_items, MF_FACTORS, MF_LR, MF_REG, SEED)
    mf.fit(tr["u"].values, tr["i"].values, tr["rating"].values, epochs=epochs, seed=SEED)
    mf.save(str(ROOT / "model_mf.npz"))

    rmse_mf = rmse(b["test"]["rating"].values,
                   mf.predict(b["test"]["u"].values, b["test"]["i"].values))
    rmse_pop = rmse(b["test"]["rating"].values,
                    np.full(len(b["test"]), float(tr["rating"].mean())))
    print(f"      RMSE  MF={rmse_mf:.4f}   全局均值基线={rmse_pop:.4f}")

    rows.append(eval_ranking("mf", lambda u, s: [i for i, _ in mf.topk(u, s, pool_n)],
                             b, rel, seen, n_items, allow))
    print(f"      mf         recall@{K}={rows[-1]['recall']:.4f} "
          f"ndcg@{K}={rows[-1]['ndcg']:.4f} coverage={rows[-1]['coverage']:.3f}")

    print("[4/5] 群体重排：随机组 vs 冲突组，mean / min / blend 对照 ...")
    rng = np.random.default_rng(SEED)
    # 两步走：原始 KNN 分数负责“召回”（保留热度带来的准确率），
    # 去掉全局热度后的分数负责“公平重排”（暴露个体口味差异）。
    S = np.vstack([knn.scores_for(u) for u in range(b["n_users"])]).astype(np.float32)
    S_ind = S - S.mean(axis=0)

    def build_groups(n_each: int = 25):
        """随机组 + 口味冲突组（成员类型偏好两两最不相似）。"""
        prof = user_profiles(b)
        sim = cosine_matrix(prof)
        groups = []
        for _ in range(n_each):
            size = int(rng.integers(3, 6))
            groups.append(("random", rng.choice(b["n_users"], size=size, replace=False).tolist()))
        for _ in range(n_each):
            size = int(rng.integers(3, 6))
            leader = int(rng.integers(0, b["n_users"]))
            members = [leader]
            while len(members) < size:
                cand = np.argsort(sim[members].max(axis=0))[: 50]   # 与现有成员最不像的前 50
                pick = int(cand[rng.integers(0, len(cand))])
                if pick not in members:
                    members.append(pick)
            groups.append(("conflict", members))
        return groups

    groups = build_groups()
    group_rows = []
    import itertools
    for fairness in ("mean", "min", "blend"):
        for kind in ("random", "conflict"):
            mins, avgs, accept, top1 = [], [], [], []
            for (gkind, members), (_, c) in zip(
                    [(k, m) for k, m in groups if k == kind],
                    itertools.cycle(DEMO_CONSTRAINTS)):
                mask = build_mask(c, b["item_genres"], b["year"], b["votes"], b["title"])
                pool = retrieve_pool(S, members, mask, 200)
                idx = np.where(pool)[0]
                if len(idx) == 0:
                    continue
                rel = pool_utility(S_ind[np.ix_(members, idx)])   # 去热度后的池内相对效用
                if fairness == "mean":
                    g = rel.mean(axis=0)
                elif fairness == "min":
                    g = rel.min(axis=0)
                else:
                    g = 0.5 * rel.mean(axis=0) + 0.5 * rel.min(axis=0)
                top = np.argsort(-g)[:K]               # 候选池内部下标
                util = rel[:, top]                     # (m, K)
                member_util = util.mean(axis=1)
                top1.append(float(util[:, 0].min()))   # 第一部片对最不满意成员的分位
                mins.append(float(member_util.min()))
                avgs.append(float(member_util.mean()))
                accept.append(1.0 if util[:, 0].min() >= 0.8 else 0.0)
            group_rows.append(dict(fairness=fairness, group=kind,
                                   top1_min=mean(top1),
                                   min_satisfaction=mean(mins),
                                   avg_satisfaction=mean(avgs),
                                   accept_rate=mean(accept)))
            r = group_rows[-1]
            print(f"      {kind:<8} {fairness:<6} 首推最差成员={r['top1_min']:.3f} "
                  f"列表最差成员={r['min_satisfaction']:.3f} "
                  f"列表平均={r['avg_satisfaction']:.3f} 接受率={r['accept_rate']:.2f}")

    print("[5/5] 写结果 ...")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary = dict(
        data=dict(users=b["n_users"], items=b["n_items"],
                  train=len(b["train"]), test=len(b["test"])),
        ranking=rows, rmse=dict(mf=rmse_mf, mean_baseline=rmse_pop),
        group=group_rows, config=dict(K=K, factors=MF_FACTORS, epochs=epochs,
                                      lr=MF_LR, reg=MF_REG, seed=SEED),
        elapsed_sec=round(time.time() - t0, 1),
    )
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n结果已写入 results/summary.json")
    print(f"总耗时 {summary['elapsed_sec']} s")


if __name__ == "__main__":
    main()
