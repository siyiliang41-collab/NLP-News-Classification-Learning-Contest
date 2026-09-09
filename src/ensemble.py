# -*- coding: utf-8 -*-
"""
多模型融合：对多个模型的预测结果做投票/加权融合，冲最高分。

原理：不同结构的模型（CNN / RNN / BERT）犯错的样本不同，投票可互补，通常比单模型高 0.5~1 个点。

输入：各模型的提交文件（每份都是两列？不，本赛题提交文件是单列 label）
      注意：本赛题提交格式只有 label 一列（0~13 的数字），没有概率（softmax）信息。
      因此采用「硬投票」：对每条样本，取所有模型预测的众数；平票时按模型列表顺序取靠前者
      （纯硬投票的多数投票，不涉及置信度）。

用法：
    python src/ensemble.py --files submit_a.csv submit_b.csv --out submit_ensemble.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from config import SUBMIT_DIR, NUM_CLASSES


def majority_vote(preds_matrix):
    """preds_matrix: (n_models, n_samples)，对每列（每条样本）取众数。

    平票规则说明（如实记录，这是硬投票的多数投票，无置信度概念）：
        `Counter.most_common(1)` 在平票时返回「第一个出现」的类别，即按模型列表顺序靠前者胜出。
        当 N 个模型对同一条样本给出 N 个不同答案时（如 3 模型各投 1 票），
        会退化成「无条件取第 1 个模型」，该样本的融合未发挥作用。
        模型越多、越接近一致时这种情况越少；需要按模型可信度打破平票时用 weighted_vote。
    """
    n_models, n_samples = preds_matrix.shape
    result = []
    for j in range(n_samples):
        votes = preds_matrix[:, j]
        cnt = Counter(votes)
        # 取票数最多的；平票时按模型列表顺序取靠前者
        result.append(cnt.most_common(1)[0][0])
    return np.array(result, dtype=int)


def weighted_vote(preds_matrix, weights):
    """加权投票：每个模型一个权重，累加各模型对每个类别的投票。"""
    n_models, n_samples = preds_matrix.shape
    weights = np.array(weights, dtype=np.float32)
    result = []
    for j in range(n_samples):
        score = np.zeros(NUM_CLASSES, dtype=np.float32)
        for m in range(n_models):
            score[preds_matrix[m, j]] += weights[m]
        result.append(int(np.argmax(score)))
    return np.array(result, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True, help="各模型提交文件路径（label 列）")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="各模型权重（可选，与 files 一一对应）")
    parser.add_argument("--out", default="submit_ensemble.csv")
    args = parser.parse_args()

    preds_list = []
    for f in args.files:
        df = pd.read_csv(f)
        preds_list.append(df["label"].values)
        print(f"[加载] {os.path.basename(f)}: {len(df)} 条, 类别范围 {df['label'].min()}~{df['label'].max()}")

    # 检查长度一致
    lengths = {len(p) for p in preds_list}
    assert len(lengths) == 1, f"各文件行数不一致: {lengths}"

    preds_matrix = np.vstack(preds_list)

    if args.weights:
        y_final = weighted_vote(preds_matrix, args.weights)
        print(f"[融合] 加权投票, 权重={args.weights}")
    else:
        y_final = majority_vote(preds_matrix)
        print(f"[融合] 多数投票, {len(preds_list)} 个模型")

    # 统计各模型一致率（两两对比第一个模型）
    for i in range(1, len(preds_list)):
        agree = (preds_list[0] == preds_list[i]).mean()
        print(f"[一致率] 模型0 vs 模型{i}: {agree:.4f}")

    out_path = os.path.join(SUBMIT_DIR, args.out)
    pd.DataFrame({"label": y_final}).to_csv(out_path, index=False)
    print(f"[输出] {out_path}, 共 {len(y_final)} 行")


if __name__ == "__main__":
    main()
