# -*- coding: utf-8 -*-
"""
数据探索分析（EDA）：产出报告需要的三类图
1. 类别分布条形图（发现类别不均衡）
2. 文本长度分布直方图（指导截断长度选择）
3. 高频 token 统计（观察匿名化字符共现）

运行：
    python src/eda.py
输出图保存到 output/figs/
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TRAIN_PATH, FIG_DIR

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(FIG_DIR, exist_ok=True)


def plot_label_dist(y, label_names):
    counts = pd.Series(y).value_counts().sort_index()
    names = [label_names[i] for i in counts.index]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, counts.values, color="#4C78A8")
    ax.set_title("训练集类别分布（14类）", fontsize=14)
    ax.set_xlabel("类别")
    ax.set_ylabel("样本数")
    ax.tick_params(axis="x", rotation=45)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_label_dist.png"), dpi=150)
    plt.close(fig)
    print("[EDA] 类别分布图已保存")


def plot_len_dist(lengths):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.clip(lengths, 0, 4000), bins=80, color="#F58518", edgecolor="white")
    ax.axvline(600, color="red", linestyle="--", label="截断阈值 600")
    ax.set_title("文本长度分布（token 数）", fontsize=14)
    ax.set_xlabel("token 数")
    ax.set_ylabel("样本数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_len_dist.png"), dpi=150)
    plt.close(fig)
    print(f"[EDA] 长度分布图已保存 (均值 {lengths.mean():.1f}, 中位数 {np.median(lengths):.0f}, "
          f"覆盖<=600占比 {(lengths <= 600).mean()*100:.1f}%)")


def plot_top_tokens(texts, topk=20):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(t.split())
    top = cnt.most_common(topk)
    tokens = [f"#{t[0]}" for t in top]
    vals = [t[1] for t in top]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(tokens[::-1], vals[::-1], color="#54A24B")
    ax.set_title(f"高频 token Top{topk}", fontsize=14)
    ax.set_xlabel("出现次数")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "eda_top_tokens.png"), dpi=150)
    plt.close(fig)
    print(f"[EDA] 高频 token 图已保存: {top[:5]}")


def main():
    print("[EDA] 读取数据（全量）...")
    df = pd.read_csv(TRAIN_PATH, sep="\t", usecols=["label", "text"])
    print(f"      共 {len(df)} 条")

    # 类别名（用官方映射）
    label_names = {v: k for k, v in {
        '科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6,
        '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13,
    }.items()}

    plot_label_dist(df["label"].values, label_names)

    lengths = df["text"].str.split().str.len().values
    plot_len_dist(lengths)

    plot_top_tokens(df["text"].values, topk=20)

    # 打印关键统计，写进报告
    print("\n[EDA] 关键统计:")
    print(f"      类别数: {df['label'].nunique()}")
    print(f"      样本最多类别: {label_names[int(df['label'].value_counts().idxmax())]} "
          f"({df['label'].value_counts().max()})")
    print(f"      样本最少类别: {label_names[int(df['label'].value_counts().idxmin())]} "
          f"({df['label'].value_counts().min()})")


if __name__ == "__main__":
    main()
