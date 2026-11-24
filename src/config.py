# -*- coding: utf-8 -*-
"""全局配置：路径与超参数。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RESULT_DIR = ROOT / "results"
CACHE_DIR = DATA_DIR / "cache"

# MovieLens latest-small：610 位用户、9742 部电影、100 836 条评分，时间跨度到 2018 年
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_SUBDIR = RAW_DIR / "ml-latest-small"

# 矩阵分解超参数
MF_FACTORS = 32
MF_EPOCHS = 30
MF_LR = 0.01
MF_REG = 0.05
SEED = 2026

# 评估
K = 10
MIN_RATINGS = 5
RELEVANT_THRESHOLD = 4.0
