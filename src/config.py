# -*- coding: utf-8 -*-
"""
全局配置：数据路径、类别映射、实验超参统一入口。
所有脚本从这里 import，避免散落的硬编码路径。

数据路径自动定位：优先环境变量 DATA_DIR，其次在常见位置自动探测各数据文件。
支持两种布局——平铺（train_set.csv 与 test_a.csv/test_b.csv 同目录）与
项目内子目录（data/train、data/test_a、data/test_b），
本地（Windows）和云端（AutoDL/阿里云PAI）都不用手动改路径。
"""
import os

# ---------- 项目根目录 ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _candidate_roots():
    """数据根目录候选，按优先级排列。"""
    roots = []
    env = os.environ.get("DATA_DIR")
    if env:
        roots.append(env)
    roots += [
        r"C:\Users\10730\Desktop\26秋小学期\data",   # 本地 Windows
        os.path.join(BASE_DIR, "data"),              # 项目内 data/
        "/mnt/workspace/TEMP-FILE-STATION",          # 阿里云 PAI DSW 网页上传默认路径
        "/mnt/workspace/data",                       # 阿里云 PAI 手动建目录
        "/mnt/workspace",                            # 阿里云 PAI 根目录
        "/root/autodl-tmp",                          # AutoDL
    ]
    return roots


def _find(name, subdirs):
    """在候选根目录 + 子目录组合中查找数据文件，返回第一个存在的路径。

    subdirs 依次尝试（如 ["", "train"] 表示先查平铺再查 train/ 子目录），
    找不到返回 None。
    """
    for root in _candidate_roots():
        for sub in subdirs:
            path = os.path.join(root, sub, name)
            if os.path.exists(path):
                return path
    return None


# 训练集与测试集可能平铺在同一目录，也可能分属 train/test_a/test_b 子目录，
# 各自独立定位；兜底指向项目默认布局，数据未就位时读取会给出清晰报错。
TRAIN_PATH = _find("train_set.csv", ["", "train"]) \
    or os.path.join(BASE_DIR, "data", "train", "train_set.csv")
TEST_A_PATH = _find("test_a.csv", ["", "test_a"]) \
    or os.path.join(BASE_DIR, "data", "test_a", "test_a.csv")
TEST_B_PATH = _find("test_b.csv", ["", "test_b"]) \
    or os.path.join(BASE_DIR, "data", "test_b", "test_b.csv")

# 指向实际训练数据所在目录（供日志/排查用，无外部依赖）
DATA_DIR = os.path.dirname(TRAIN_PATH)

# ---------- 项目内输出目录 ----------
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
