# -*- coding: utf-8 -*-
"""生成报告用的图：Recall@K 曲线、公平-效率权衡、去热度消融。

    python eval/figures.py
输出：results/figs/*.png 与 results/figures.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))   # 避免写用户目录失败

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import data as D                                    # noqa: E402
from src.baselines import ItemKNN, Popular                   # noqa: E402
from src.config import K, RESULT_DIR, SEED                   # noqa: E402
from src.constraints import Constraints                      # noqa: E402
from src.metrics import mean, ndcg_at_k, recall_at_k        # noqa: E402
from src.mf import MF                                        # noqa: E402
from src.rank import pool_utility, retrieve_pool             # noqa: E402

FIG_DIR = RESULT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体
for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

KS = [5, 10, 20, 50]
GROUP_CONSTRAINTS = [
    Constraints(n_people=3, exclude_genres={"Horror"}),
    Constraints(n_people=3, include_genres={"Comedy"}),
    Constraints(n_people=3, year_min=2010),
    Constraints(n_people=4, include_genres={"Animation"}, min_votes=50),
]


def recall_curve(score_fn, b, rel, seen, allow):
    out = {k: [] for k in KS}
    for u in range(b["n_users"]):
        if not rel[u]:
            continue
        recs = [i for i in score_fn(u) if allow[i]][: max(KS)]
        for k in KS:
            out[k].append(recall_at_k(recs, rel[u], k))
    return {k: mean(v) for k, v in out.items()}


def build_groups(n_users, rng, sim, n_each=25):
    random_g, conflict_g = [], []
    for _ in range(n_each):
        size = int(rng.integers(3, 6))
        random_g.append(rng.choice(n_users, size=size, replace=False).tolist())
    for _ in range(n_each):
        size = int(rng.integers(3, 6))
        mem = [int(rng.integers(0, n_users))]
        while len(mem) < size:
            cand = np.argsort(sim[mem].max(axis=0))[:30]
            pick = int(cand[rng.integers(0, len(cand))])
            if pick not in mem:
                mem.append(pick)
        conflict_g.append(mem)
    return {"random": random_g, "conflict": conflict_g}


def group_eval(b, S, S_ind, groups, rng, use_ind=True):
    res = {}
    for kind, gs in groups.items():
        for fair in ("mean", "min"):
            top1, avg = [], []
            for gi, members in enumerate(gs):
                c = GROUP_CONSTRAINTS[gi % len(GROUP_CONSTRAINTS)]
                mask = (np.ones(b["n_items"], dtype=bool))
                from src.rank import build_mask
                mask = build_mask(c, b["item_genres"], b["year"], b["votes"], b["title"])
                pool = retrieve_pool(S, members, mask, 200)
                idx = np.where(pool)[0]
                if len(idx) == 0:
                    continue
                mat = S_ind if use_ind else S
                rel = pool_utility(mat[np.ix_(members, idx)])
                g = rel.mean(axis=0) if fair == "mean" else rel.min(axis=0)
                top = np.argsort(-g)[:K]
                util = rel[:, top]
                top1.append(float(util[:, 0].min()))
                avg.append(float(util.mean(axis=1).mean()))
            res[(kind, fair)] = (mean(top1), mean(avg))
    return res


def main():
    b = D.build()
    n_users, n_items = b["n_users"], b["n_items"]
    rel = D.relevant_by_user(b["test"], n_users, b["threshold"])
    seen = D.seen_by_user(b["train"], n_users)
    allow = b["votes"] >= 20

    pop = Popular(b["train"], n_items)
    knn = ItemKNN(D.rating_matrix(b["train"], n_users, n_items))
    mf = MF.load(str(ROOT / "model_mf.npz"))

    curves = {
        "热门榜": recall_curve(lambda u: pop.topk(seen[u], max(KS)), b, rel, seen, allow),
        "ItemKNN": recall_curve(lambda u: knn.topk(u, seen[u], max(KS)), b, rel, seen, allow),
        "矩阵分解": recall_curve(lambda u: [i for i, _ in mf.topk(u, seen[u], max(KS))],
                             b, rel, seen, allow),
    }

    # 图 1：Recall@K 曲线
    plt.figure(figsize=(6.2, 3.8), dpi=200)
    for name, c in curves.items():
        plt.plot(KS, [c[k] for k in KS], marker="o", label=name)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("三种召回策略的 Recall@K（留一法，363 位用户）")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_recall_k.png")
    plt.close()

    # 群体重排所需矩阵
    S = np.vstack([knn.scores_for(u) for u in range(n_users)]).astype(np.float32)
    S_ind = S - S.mean(axis=0)
    rng = np.random.default_rng(SEED)
    prof = np.zeros((n_users, b["genre"].shape[1]), dtype=np.float32)
    cnt = np.zeros(n_users, dtype=np.float32)
    for u, i, r in zip(b["train"]["u"].values, b["train"]["i"].values,
                       b["train"]["rating"].values):
        prof[u] += b["genre"][i] * (r / 5.0)
        cnt[u] += 1
    cnt[cnt == 0] = 1
    prof /= cnt[:, None]
    norm = np.linalg.norm(prof, axis=1)
    norm[norm == 0] = 1e-9
    sim = (prof / norm[:, None]) @ (prof / norm[:, None]).T

    groups = build_groups(n_users, rng, sim)
    res_ind = group_eval(b, S, S_ind, groups, rng, use_ind=True)
    res_raw = group_eval(b, S, S_ind, groups, rng, use_ind=False)

    # 图 2：公平-效率权衡
    labels = ["随机组·mean", "随机组·min", "冲突组·mean", "冲突组·min"]
    keys = [("random", "mean"), ("random", "min"), ("conflict", "mean"), ("conflict", "min")]
    x = np.arange(len(labels))
    worst = [res_ind[k][0] for k in keys]
    aver = [res_ind[k][1] for k in keys]
    plt.figure(figsize=(6.6, 3.8), dpi=200)
    w = 0.36
    plt.bar(x - w / 2, worst, w, label="首推·最差成员效用")
    plt.bar(x + w / 2, aver, w, label="列表·平均效用")
    plt.xticks(x, labels)
    plt.ylim(0, 1.12)
    plt.ylabel("效用（池内分位）")
    plt.title("max-min 重排的效果：最差成员↑、平均↓")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(loc="upper left", ncol=2, fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_fairness_tradeoff.png")
    plt.close()

    # 图 3：去热度消融
    plt.figure(figsize=(6.2, 3.6), dpi=200)
    x = np.arange(2)
    raw = [res_raw[("random", "mean")][0], res_raw[("random", "min")][0]]
    ind = [res_ind[("random", "mean")][0], res_ind[("random", "min")][0]]
    plt.bar(x - 0.18, raw, 0.36, label="原始 KNN 分数")
    plt.bar(x + 0.18, ind, 0.36, label="去掉全局热度后")
    plt.xticks(x, ["mean 聚合", "min 聚合"])
    plt.ylim(0, 1)
    plt.ylabel("首推·最差成员效用")
    plt.title("消融：不去热度时，公平策略几乎没有区别")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_ablation.png")
    plt.close()

    out = dict(
        recall_curves=curves,
        group_ind={f"{k[0]}_{k[1]}": v for k, v in res_ind.items()},
        group_raw={f"{k[0]}_{k[1]}": v for k, v in res_raw.items()},
    )
    (RESULT_DIR / "figures.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, c in curves.items():
        print(f"{name:<8} " + " ".join(f"R@{k}={c[k]:.4f}" for k in KS))
    for k, v in res_ind.items():
        print(f"{k[0]:<8} {k[1]:<5} 首推最差={v[0]:.3f} 列表平均={v[1]:.3f}")
    print("图已写入", FIG_DIR)


if __name__ == "__main__":
    main()
