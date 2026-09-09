# -*- coding: utf-8 -*-
"""
性能迭代可视化：读取 docs/scores.csv 分数台账，画成柱状图/折线图。
这是设计报告评分表的硬性要求（"性能迭代过程的可视化计算界面和数据说明"）。

用法：
    python src/plot_scores.py
输出图保存到 output/figs/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SCORES_CSV, FIG_DIR

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(FIG_DIR, exist_ok=True)

# 模型显示名映射（按技术路线顺序）
MODEL_ORDER = [
    ("tfidf_lr", "TF-IDF + LR"),
    ("word2vec_textcnn", "TextCNN"),
    ("word2vec_bilstm_attn", "BiLSTM+Attn"),
    ("bert", "BERT"),
]
# 取每个模型的最好一次实验
def _best(df):
    rows = []
    for model, name in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        best = sub.loc[sub["cv_f1"].astype(float).idxmax()]
        rows.append({"model": model, "name": name, "cv_f1": float(best["cv_f1"])})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(SCORES_CSV)
    best = _best(df)
    if best.empty:
        print("台账中还没有可可视化的分数。")
        return

    print("[可视化] 各模型最优 macro F1:")
    print(best.to_string(index=False))

    # 柱状图
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][:len(best)]
    bars = ax.bar(best["name"], best["cv_f1"], color=colors, width=0.5)
    for b, v in zip(bars, best["cv_f1"]):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro F1")
    ax.set_title("模型性能迭代对比", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "score_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"[可视化] 柱状图已保存: {os.path.join(FIG_DIR, 'score_comparison.png')}")

    # 折线图（迭代上升趋势）
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best["name"], best["cv_f1"], marker="o", linewidth=2, color="#4C78A8")
    for i, (name, v) in enumerate(zip(best["name"], best["cv_f1"])):
        ax.annotate(f"{v:.4f}", (name, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("macro F1")
    ax.set_title("性能迭代上升曲线", fontsize=14)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "score_trend.png"), dpi=150)
    plt.close(fig)
    print(f"[可视化] 折线图已保存: {os.path.join(FIG_DIR, 'score_trend.png')}")


if __name__ == "__main__":
    main()
