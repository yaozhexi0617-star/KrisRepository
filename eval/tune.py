# -*- coding: utf-8 -*-
"""超参数与冷门过滤的调参脚本。

    python eval/tune.py --factors 16 --epochs 20 --lr 0.005 --reg 0.1 --min-votes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data as D                                   # noqa: E402
from src.config import K, RELEVANT_THRESHOLD, SEED          # noqa: E402
from src.metrics import catalog_coverage, mean, ndcg_at_k, recall_at_k, rmse  # noqa: E402
from src.mf import MF                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--min-votes", type=int, default=10,
                    help="只推荐训练集中评分人数 >= 该值的条目")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    b = D.build()
    tr = b["train"]
    votes = b["votes"]
    allow = votes >= args.min_votes

    mf = MF(b["n_users"], b["n_items"], args.factors, args.lr, args.reg, SEED)
    if args.quiet:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            mf.fit(tr["u"].values, tr["i"].values, tr["rating"].values,
                   epochs=args.epochs, seed=SEED)
    else:
        mf.fit(tr["u"].values, tr["i"].values, tr["rating"].values,
               epochs=args.epochs, seed=SEED)

    rel = D.relevant_by_user(b["test"], b["n_users"], RELEVANT_THRESHOLD)
    seen = D.seen_by_user(b["train"], b["n_users"])
    recs, rr, nn = [], [], []
    for u in range(b["n_users"]):
        if not rel[u]:
            continue
        r = [i for i, _ in mf.topk(u, seen[u], K, mask=allow)]
        recs.append(r)
        rr.append(recall_at_k(r, rel[u], K))
        nn.append(ndcg_at_k(r, rel[u], K))

    test_r = rmse(b["test"]["rating"].values,
                  mf.predict(b["test"]["u"].values, b["test"]["i"].values))
    print(f"factors={args.factors} epochs={args.epochs} lr={args.lr} reg={args.reg} "
          f"min_votes={args.min_votes} | recall@{K}={mean(rr):.4f} ndcg@{K}={mean(nn):.4f} "
          f"rmse={test_r:.4f} coverage={catalog_coverage(recs, b['n_items']):.3f}")


if __name__ == "__main__":
    main()
