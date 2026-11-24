# -*- coding: utf-8 -*-
"""生成推荐理由：只用模型真实用到的信息（类型、成员在候选池内的相对位置）。"""
from __future__ import annotations


def _pct(x: float) -> str:
    rank = max(1, int(round((1 - x) * 100)))
    return f"前 {rank}%"


def explain(rank_no: int, title: str, year: int, genres: set[str],
            utils: list[float], names: list[str], fairness: str = "min") -> str:
    g = "/".join(sorted(genres)) if genres else "未标注类型"
    head = f"{rank_no}. 《{title}》({year})　[{g}]"

    lo = min(range(len(utils)), key=lambda k: utils[k])
    hi = max(range(len(utils)), key=lambda k: utils[k])
    if fairness == "min":
        reason = (f"顾虑最多的是 {names[lo]}（在他的候选池里 {_pct(utils[lo])}），"
                  f"{names[hi]} 则 {_pct(utils[hi])}。")
    else:
        reason = (f"{names[hi]} 最想要（{_pct(utils[hi])}），"
                  f"{names[lo]} 相对靠后（{_pct(utils[lo])}）。")
    return f"{head}　{reason}"


def one_line(title: str, year: int, genres: set[str], utils: list[float]) -> str:
    g = "/".join(sorted(genres)) if genres else "-"
    u = " ".join(f"{x:.2f}" for x in utils)
    return f"《{title}》({year}) [{g}] 成员效用: {u}"
