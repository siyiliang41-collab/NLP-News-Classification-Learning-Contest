# -*- coding: utf-8 -*-
"""
全局配置：数据路径、类别映射、实验超参统一入口。
所有脚本从这里 import，避免散落的硬编码路径。
"""
import os

# ---------- 数据路径（数据在 26秋小学期/data 下）----------
DATA_DIR = r"C:\Users\10730\Desktop\26秋小学期\data"

TRAIN_PATH = os.path.join(DATA_DIR, "train_set.csv")
TEST_A_PATH = os.path.join(DATA_DIR, "test_a.csv")
TEST_B_PATH = os.path.join(DATA_DIR, "test_b.csv")

# ---------- 项目内输出目录 ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUBMIT_DIR = os.path.join(BASE_DIR, "data", "submit")
SCORES_CSV = os.path.join(BASE_DIR, "docs", "scores.csv")

# ---------- 类别映射（官方固定，勿改动）----------
LABEL_MAP = {
    '科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6,
    '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_CLASSES = len(LABEL_MAP)

# ---------- 通用超参 ----------
MAX_SEQ_LEN = 600   # 文本截断长度（token 数），覆盖约 50% 样本；TextCNN 可加大
N_FOLDS = 5          # 交叉验证折数
SEED = 42
