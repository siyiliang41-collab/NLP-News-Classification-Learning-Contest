# -*- coding: utf-8 -*-
"""
类别不均衡实验可视化：CE vs 类别加权 vs Focal Loss 的各类别 F1 对比。

实验结论（textcnn_loss.py --holdout, epochs=3, CBOW 词向量）：
    CE 在「大但易混淆」的新闻域(社会/时政/房产)更强；
    Focal 在「小但主题鲜明」的类(星座/彩票/时尚)更强；
    加权/Focal 整体 macro F1 反而低于 CE —— 本数据集瓶颈是「类间混淆」而非「类别不均衡」。

产出两张图：
    1. loss_ablation_delta.png   发散条形图（focal − CE 各类别差值，正值蓝/负值红）
    2. loss_ablation_grouped.png 分组条形图（3 种损失 × 14 类 F1）

运行：
    python src/plot_loss_ablation.py
输出图保存到 output/figs/
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIG_DIR

# 中文字体（与 eda.py 保持一致）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

os.makedirs(FIG_DIR, exist_ok=True)

# ---------- 配色（dataviz 参考调色板：蓝=正向/焦点，红=负向，灰=基线，青=加权）----------
BLUE = "#2a78d6"    # Focal / 正向差值
RED = "#e34948"     # 负向差值
AQUA = "#1baf7a"    # weighted
GRAY = "#898781"    # CE 基线（去强调）
INK = "#52514e"     # 次级文字

# 类别顺序（按 label 0~13）
LABELS = ['科技', '股票', '体育', '娱乐', '时政', '社会', '教育',
          '财经', '家居', '游戏', '房产', '时尚', '彩票', '星座']

CE = {
    '科技': 0.9268, '股票': 0.9295, '体育': 0.9818, '娱乐': 0.9437,
    '时政': 0.8826, '社会': 0.8444, '教育': 0.9143, '财经': 0.8079,
    '家居': 0.9043, '游戏': 0.9063, '房产': 0.9079, '时尚': 0.8847,
    '彩票': 0.9117, '星座': 0.9133,
}
WEIGHTED = {
    '科技': 0.9071, '股票': 0.9152, '体育': 0.9777, '娱乐': 0.9127,
    '时政': 0.8537, '社会': 0.7632, '教育': 0.8929, '财经': 0.7856,
    '家居': 0.8952, '游戏': 0.8822, '房产': 0.8766, '时尚': 0.8287,
    '彩票': 0.8475, '星座': 0.9119,
}
FOCAL = {
    '科技': 0.9057, '股票': 0.9144, '体育': 0.9825, '娱乐': 0.9297,
    '时政': 0.8585, '社会': 0.7758, '教育': 0.8944, '财经': 0.7929,
    '家居': 0.8895, '游戏': 0.8831, '房产': 0.8332, '时尚': 0.8920,
    '彩票': 0.9315, '星座': 0.9565,
}


def plot_delta():
    """发散条形图：Focal − CE 的各类别差值，正值蓝(帮到)/负值红(伤到)。"""
    delta = {k: FOCAL[k] - CE[k] for k in LABELS}
    items = sorted(delta.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [BLUE if v >= 0 else RED for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, vals, color=colors, height=0.62)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlim(-0.10, 0.06)
    for y, v in enumerate(vals):
        ax.text(v + (0.001 if v >= 0 else -0.001), y, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK)
    ax.set_xlabel("Focal - CE 的 F1 差值（>0 表示 Focal 更优）", fontsize=11)
    ax.set_title("Focal Loss 相对 CE 的各类别增益", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "loss_ablation_delta.png"), dpi=150)
    plt.close(fig)
    print("[可视化] 发散差值图已保存: loss_ablation_delta.png")


def plot_grouped():
    """分组条形图：3 种损失 × 14 类 F1。CE 用灰(基线)，weighted 青、Focal 蓝。"""
    ce_vals = [CE[k] for k in LABELS]
    w_vals = [WEIGHTED[k] for k in LABELS]
    f_vals = [FOCAL[k] for k in LABELS]
    x = range(len(LABELS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar([i - w for i in x], ce_vals, w, label="CE（交叉熵）", color=GRAY)
    ax.bar(x, w_vals, w, label="加权 CE", color=AQUA)
    ax.bar([i + w for i in x], f_vals, w, label="Focal Loss", color=BLUE)
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_ylim(0.70, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("三种损失函数下各类别 F1 对比", fontsize=13)
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "loss_ablation_grouped.png"), dpi=150)
    plt.close(fig)
    print("[可视化] 分组对比图已保存: loss_ablation_grouped.png")


if __name__ == "__main__":
    plot_delta()
    plot_grouped()
