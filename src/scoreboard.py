# -*- coding: utf-8 -*-
"""
分数记录台账：所有实验的 CV macro F1 统一写进 docs/scores.csv，
供后面的「性能迭代可视化」直接读取绘图。

字段说明：
    model      模型名（如 tfidf_lr / fasttext / textcnn / bilstm_attn / bert）
    detail     变体/技巧说明（如 baseline / 加对抗训练 / 加融合）
    cv_f1      K 折交叉验证 macro F1 均值
    std        各折标准差
    note       备注
"""
import os
import csv
import pandas as pd

from config import SCORES_CSV

_COLUMNS = ["model", "detail", "cv_f1", "std", "note"]


def record(model, detail, cv_f1, std=None, note=""):
    """追加一条实验记录。cv_f1 为 float，std 可选。"""
    os.makedirs(os.path.dirname(SCORES_CSV), exist_ok=True)
    row = {
        "model": model,
        "detail": detail,
        "cv_f1": f"{cv_f1:.4f}",
        "std": "" if std is None else f"{std:.4f}",
        "note": note,
    }
    # 若文件已存在则追加，否则写表头
    file_exists = os.path.exists(SCORES_CSV)
    with open(SCORES_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[记分] {model} ({detail}): cv_f1 = {row['cv_f1']} -> {SCORES_CSV}")


def show():
    """打印当前台账。"""
    if not os.path.exists(SCORES_CSV):
        print("台账为空，尚未记录任何分数。")
        return
    df = pd.read_csv(SCORES_CSV)
    print(df.to_string(index=False))


if __name__ == "__main__":
    show()
