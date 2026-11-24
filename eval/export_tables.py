# -*- coding: utf-8 -*-
"""把 results/summary.json 导出成 CSV，方便写报告和答辩时直接引用。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUM = ROOT / "results" / "summary.json"

data = json.loads(SUM.read_text(encoding="utf-8"))

rows = data["ranking"]
rank_csv = ROOT / "results" / "ranking.csv"
with open(rank_csv, "w", encoding="utf-8-sig", newline="") as f:
    f.write("model,recall@10,ndcg@10,coverage,n_users\n")
    for r in rows:
        f.write(f"{r['model']},{r['recall']:.4f},{r['ndcg']:.4f},"
                f"{r['coverage']:.4f},{r['n_users']}\n")
    f.write(f"mean_baseline_rmse,,{data['rmse']['mean_baseline']:.4f},,\n")
    f.write(f"mf_rmse,,{data['rmse']['mf']:.4f},,\n")

grp_csv = ROOT / "results" / "group.csv"
with open(grp_csv, "w", encoding="utf-8-sig", newline="") as f:
    f.write("group,fairness,top1_min,list_min,list_avg,accept_rate\n")
    for g in data["group"]:
        f.write(f"{g['group']},{g['fairness']},{g['top1_min']:.4f},"
                f"{g['min_satisfaction']:.4f},{g['avg_satisfaction']:.4f},"
                f"{g['accept_rate']:.2f}\n")

print("wrote", rank_csv)
print("wrote", grp_csv)
