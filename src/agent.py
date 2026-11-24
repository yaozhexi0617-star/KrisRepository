# -*- coding: utf-8 -*-
"""「今晚看什么」Agent：多轮对话 + 硬约束过滤 + 群体重排。

打分分两步：
  1) ItemKNN 原始分数负责召回（保留“热门片更可能被喜欢”的准确率优势）；
  2) 去掉全局热度后的分数负责重排（否则所有人的榜单高度相似，群体公平无从谈起）。

用法：
    python -m src.agent --demo                 # 脚本化演示（便于录屏）
    python -m src.agent --users 1,2,3          # 用数据集里真实的用户模拟小队
    python -m src.agent --interactive          # 自己输指令
"""
from __future__ import annotations

import argparse
import os
import pickle
import random

import numpy as np

from . import data as D
from .config import SEED
from .constraints import Constraints, parse
from .explain import explain
from .rank import build_mask, pool_utility, retrieve_pool

CACHE = "agent_cache.pkl"


class MovieAgent:
    def __init__(self, b, S, S_ind, members, fairness="min", names=None):
        self.b = b
        self.S = S                    # (n_users, n_items) KNN 原始分
        self.S_ind = S_ind            # 去掉全局热度后的分数
        self.members = members
        self.fairness = fairness
        self.names = names or [f"成员{i + 1}" for i in range(len(members))]
        self.c = Constraints(n_people=len(members))
        self.ban: set[int] = set()
        self.shown: set[int] = set()
        self.turn = 0
        self.verbose = False

    # ------------------------------------------------------------------
    def recommend(self, k: int = 3) -> list[dict]:
        self.turn += 1
        b = self.b
        mask = build_mask(self.c, b["item_genres"], b["year"], b["votes"],
                          b["title"], ban=self.ban)
        if self.verbose:
            print(f"   ├─ 约束对象：{self.c.describe()}")
            print(f"   ├─ 满足硬约束的条目：{int(mask.sum())} / {b['n_items']}")
        pool = retrieve_pool(self.S, self.members, mask, 200)
        idx = np.where(pool)[0]
        if len(idx) == 0:
            return []
        if self.verbose:
            print(f"   ├─ 候选池：{len(idx)} 部（每位成员合法条目内 Top-200 的并集）")

        rel = pool_utility(self.S_ind[np.ix_(self.members, idx)])
        if self.fairness == "mean":
            g = rel.mean(axis=0)
        elif self.fairness == "min":
            g = rel.min(axis=0)
        else:
            g = 0.5 * rel.mean(axis=0) + 0.5 * rel.min(axis=0)
        if self.verbose:
            print(f"   └─ 群体重排：{self.fairness}（去热度后池内分位，"
                  f"{len(self.members)} 位成员）")

        order = np.argsort(-g)
        out = []
        for pos, col in enumerate(order[:k], start=1):
            item = int(idx[col])
            if item in self.ban or item in self.shown:
                continue
            utils = [float(x) for x in rel[:, col]]
            out.append(dict(
                item=item,
                title=str(b["title"][item]),
                year=int(b["year"][item]),
                genres=b["item_genres"][item],
                utils=utils,
                text=explain(pos, str(b["title"][item]), int(b["year"][item]),
                             b["item_genres"][item], utils, self.names, self.fairness),
            ))
            self.shown.add(item)
        return out

    def say(self, text: str) -> Constraints:
        self.c = parse(text, self.c)
        if self.c.n_people != len(self.members):
            print(f"   （提示：本次会话固定为 {len(self.members)} 位成员，"
                  f"“{self.c.n_people} 个人”仅用于演示约束解析）")
        return self.c

    def exclude(self, item: int):
        self.ban.add(int(item))
        self.shown.discard(int(item))

    def refresh(self):
        """“换一批”：把已展示过的片子排除在本轮之外。"""
        self.ban.update(self.shown)
        self.shown.clear()

    def state(self) -> str:
        return (f"约束：{self.c.describe()}｜公平策略：{self.fairness}｜"
                f"已排除 {len(self.ban)} 部｜第 {self.turn} 轮")


# ----------------------------------------------------------------------
def build_scorer(b):
    """构建 (S, S_ind) 打分矩阵，并缓存到磁盘，避免每次启动都重算。"""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            obj = pickle.load(f)
        if obj.get("n_users") == b["n_users"] and obj.get("n_items") == b["n_items"]:
            return obj["S"], obj["S_ind"]

    from .baselines import ItemKNN
    R = D.rating_matrix(b["train"], b["n_users"], b["n_items"])
    knn = ItemKNN(R)
    S = np.vstack([knn.scores_for(u) for u in range(b["n_users"])]).astype(np.float32)
    S_ind = (S - S.mean(axis=0)).astype(np.float32)
    with open(CACHE, "wb") as f:
        pickle.dump(dict(S=S, S_ind=S_ind, n_users=b["n_users"],
                         n_items=b["n_items"]), f)
    return S, S_ind


def load(users=None, fairness="blend"):
    b = D.build()
    if not users:
        users = sorted(random.Random(SEED).sample(range(b["n_users"]), 4))
    members = [b["u_index"][u] for u in users if u in b["u_index"]] or [0, 1, 2]
    S, S_ind = build_scorer(b)
    names = [f"用户{u}" for u in users] if users else None
    return MovieAgent(b, S, S_ind, members, fairness, names)


DEMO_SCRIPT = [
    "我们4个人，想看点轻松的喜剧，不要恐怖片",
    "太老的不要，2010年以后的",
    "换一批",
]


def run_demo(agent: MovieAgent):
    print("=" * 70)
    print("演示：小队选片对话（脚本化）")
    print("=" * 70)
    for text in DEMO_SCRIPT:
        print(f"\n[用户] {text}")
        agent.say(text)
        print(f"[Agent] 解析 → {agent.c.describe()}")
        if text == "换一批":
            agent.refresh()
        opts = agent.recommend(3)
        if not opts:
            print("[Agent] 没有同时满足条件的片子，建议放宽类型或年代。")
            continue
        for o in opts:
            print("   " + o["text"])
    print("\n[状态]", agent.state())


def run_interactive(agent: MovieAgent):
    print("输入约束（例：3个人，不要恐怖片，2010年以后）；q 退出；直接回车=换一批。")
    print("当前：", agent.state())
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"q", "quit", "exit"}:
            break
        if text:
            agent.say(text)
        agent.refresh()
        opts = agent.recommend(3)
        if not opts:
            print("没有满足条件的候选，试试放宽约束。")
            continue
        for o in opts:
            print("  " + o["text"])
        print("  [" + agent.state() + "]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--users", type=str, default="")
    ap.add_argument("--fairness", default="blend", choices=["min", "mean", "blend"])
    ap.add_argument("--verbose", action="store_true", help="打印中间步骤，适合录屏")
    args = ap.parse_args()

    users = [int(x) for x in args.users.split(",") if x.strip()] if args.users else None
    agent = load(users, args.fairness)
    agent.verbose = args.verbose
    if args.interactive:
        run_interactive(agent)
    else:
        run_demo(agent)


if __name__ == "__main__":
    main()
