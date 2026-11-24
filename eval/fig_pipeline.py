# -*- coding: utf-8 -*-
"""画系统流程图：results/figs/fig0_pipeline.png"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [f]
        break
plt.rcParams["axes.unicode_minus"] = False

FIG = ROOT / "results" / "figs"
FIG.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(6.4, 7.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 13.0)
ax.axis("off")

STEP = "#dbe7f3"
ASSET = "#e8f2e2"
ACCENT = "#f6e2d3"


def box(x, y, w, h, text, color=STEP, size=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                linewidth=1.0, edgecolor="#4a5a6a", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, linespacing=1.45,
            fontweight="bold" if bold else "normal")


def arrow(x1, y1, x2, y2, style="-|>", color="#4a5a6a"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=1.0, color=color))


# 主流程（自上而下）
box(1.6, 11.2, 6.8, 0.95, "用户：一句话说清需求\n“我们 4 个人，点轻松的喜剧，别太老，不要恐怖片”",
    color=ACCENT, size=8.5, bold=True)

box(1.6, 9.7, 6.8, 0.95, "① 约束解析\n词典 + 否定作用域 + 年代正则 → Constraints 对象")
box(1.6, 8.2, 6.8, 0.95, "② 候选召回\n每位成员在合法条目中取 Top-200，取并集")
box(1.6, 6.7, 6.8, 0.95, "③ 硬约束过滤\n类型 / 年代 / 热度 / 已看，编译成布尔掩码")
box(1.6, 5.2, 6.8, 0.95, "④ 群体重排\n去掉全局热度 → 池内分位效用 → max-min / blend")
box(1.6, 3.7, 6.8, 0.95, "⑤ 解释生成\n“谁最想要、谁最顾虑”，只引用模型用到的量")
box(1.6, 2.2, 6.8, 0.95, "⑥ 展示 Top-3 与推荐理由", color="#f3ecd8", bold=True)

# 反馈回路：从“展示”回到“群体重排”，右侧绕一圈
arrow(8.4, 2.2 + 0.48, 9.2, 2.2 + 0.48)
arrow(9.2, 2.2 + 0.48, 9.2, 5.2 + 0.48)
arrow(9.2, 5.2 + 0.48, 8.4, 5.2 + 0.48)
ax.text(9.34, 3.9, "换一批 / 排除\n更新 ban、shown 后重新重排", fontsize=6.8,
        va="center", ha="left", rotation=90, color="#4a5a6a")

for y in (9.7, 8.2, 6.7, 5.2, 3.7):
    arrow(5.0, y + 0.95, 5.0, y + 1.2)

# 右侧资产
box(1.6, 0.35, 6.8, 1.15,
    "数据与模型：MovieLens latest-small（610 用户 / 9724 片）\n"
    "ItemKNN 打分缓存 · 矩阵分解作为评分对照 · results/summary.json",
    color=ASSET, size=7.8)
arrow(5.0, 1.5, 5.0, 2.2, style="-|>", color="#7a8a9a")

ax.text(5.0, 12.62, "「今晚看什么」Agent 的处理流程", ha="center",
        fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(FIG / "fig0_pipeline.png", bbox_inches="tight")
print("saved", FIG / "fig0_pipeline.png")
