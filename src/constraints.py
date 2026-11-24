# -*- coding: utf-8 -*-
"""把一句中文/英文口语指令解析成可执行的硬软约束。

设计取舍：这里用“词典 + 正则 + 否定词作用域”的规则，而不是训练一个分类器。
原因是约束解析的错误代价很高（解析错一条约束，整轮推荐作废），规则的可解释性和
可控性更好；而本项目真正需要学习的是“这群人喜欢什么”，交给矩阵分解和聚类。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GENRE_WORDS = {
    "动作": "Action", "打斗": "Action", "action": "Action",
    "冒险": "Adventure", "adventure": "Adventure",
    "动画": "Animation", "动漫": "Animation", "animation": "Animation",
    "儿童": "Children", "亲子": "Children", "children": "Children",
    "喜剧": "Comedy", "搞笑": "Comedy", "comedy": "Comedy",
    "犯罪": "Crime", "crime": "Crime",
    "纪录": "Documentary", "纪录片": "Documentary", "documentary": "Documentary",
    "剧情": "Drama", "drama": "Drama",
    "奇幻": "Fantasy", "fantasy": "Fantasy",
    "黑色": "Film-Noir", "黑色电影": "Film-Noir",
    "恐怖": "Horror", "吓人": "Horror", "惊悚片": "Horror", "horror": "Horror",
    "音乐": "Musical", "歌舞": "Musical", "musical": "Musical",
    "悬疑": "Mystery", "推理": "Mystery", "mystery": "Mystery",
    "爱情": "Romance", "romance": "Romance", "恋爱": "Romance",
    "科幻": "Sci-Fi", "scifi": "Sci-Fi",
    "惊悚": "Thriller", "thriller": "Thriller",
    "战争": "War", "war": "War",
    "西部": "Western", "western": "Western",
}

NEG_WORDS = ("不要", "不想", "别看", "别放", "避开", "除了", "不看", "no ", "not ", "without")

YEAR_PATTERNS = [
    (r"(近|最近)\s*([0-9一二三五])?\s*年", "recent"),
    (r"(9[0-9])\s*年代", "90s"),
    (r"(20[0-9]{2})\s*年(?:以后|之后)", "since"),
    (r"(19[0-9]{2})\s*年(?:以前|之前)", "before"),
]

CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
          "七": 7, "八": 8, "九": 9, "十": 10}


@dataclass
class Constraints:
    n_people: int = 1
    include_genres: set[str] = field(default_factory=set)
    exclude_genres: set[str] = field(default_factory=set)
    year_min: int = 0
    year_max: int = 9999
    min_votes: int = 0            # 热度下限：评分人数少于该值的片视为太冷门
    exclude_titles: list[str] = field(default_factory=list)
    max_runtime: int = 0          # 预留：数据集中没有片长时忽略

    def describe(self) -> str:
        parts = []
        if self.n_people > 1:
            parts.append(f"{self.n_people} 人一起看")
        if self.include_genres:
            parts.append("类型包含 " + "/".join(sorted(self.include_genres)))
        if self.exclude_genres:
            parts.append("排除 " + "/".join(sorted(self.exclude_genres)))
        if self.year_min:
            parts.append(f"{self.year_min} 年以后")
        if self.year_max < 9999:
            parts.append(f"{self.year_max} 年以前")
        if self.min_votes:
            parts.append(f"评分人数 ≥ {self.min_votes}")
        if self.exclude_titles:
            parts.append("不要 " + "、".join(self.exclude_titles))
        return "；".join(parts) if parts else "无特殊约束"

    def is_empty(self) -> bool:
        return not (self.include_genres or self.exclude_genres or self.year_min
                    or self.year_max < 9999 or self.min_votes
                    or self.exclude_titles)


def parse(text: str, base: Constraints | None = None) -> Constraints:
    """在 base 的基础上叠加新指令（多轮对话）。"""
    c = Constraints(**{k: (set(v) if isinstance(v, set) else list(v) if isinstance(v, list) else v)
                       for k, v in vars(base).items()}) if base else Constraints()
    s = " " + text.lower() + " "

    # 人数
    m = re.search(r"(\d+|[" + "".join(CN_NUM) + r"])\s*(个)?人", text)
    if m:
        tok = m.group(1)
        c.n_people = int(tok) if tok.isdigit() else CN_NUM.get(tok, 1)

    # 类型：先按出现位置切分“不要/除了”的作用域
    neg_spans = []
    for w in NEG_WORDS:
        for mm in re.finditer(re.escape(w), s):
            neg_spans.append((mm.start(), mm.start() + 8))   # 否定词后 8 个字符内算否定域

    for word, genre in GENRE_WORDS.items():
        for mm in re.finditer(re.escape(word), s):
            negated = any(a <= mm.start() < b for a, b in neg_spans)
            if negated:
                c.exclude_genres.add(genre)
                c.include_genres.discard(genre)
            else:
                c.include_genres.add(genre)
                c.exclude_genres.discard(genre)

    # 年代
    if re.search(r"(近|最近)\s*([0-9一二三五])?\s*年", text):
        num = re.search(r"(近|最近)\s*([0-9一二三五])?\s*年", text).group(2)
        years = int(num) if num and num.isdigit() else 5
        c.year_min = 2018 - years
    if re.search(r"(9[0-9])\s*年代", text):
        c.year_min, c.year_max = 1990, 1999
    m = re.search(r"(20[0-9]{2})\s*年(?:以后|之后)", text)
    if m:
        c.year_min = int(m.group(1))
    m = re.search(r"(19[0-9]{2})\s*年(?:以前|之前)", text)
    if m:
        c.year_max = int(m.group(1))

    # 热度："热门" / "小众"
    if re.search(r"(热门|经典|大家都看)", text):
        c.min_votes = 50
    if re.search(r"(小众|冷门|没看过)", text):
        c.min_votes = 0

    return c


def add_exclusion(c: Constraints, title: str) -> Constraints:
    c.exclude_titles = list(dict.fromkeys(c.exclude_titles + [title]))
    return c
