# -*- coding: utf-8 -*-
"""
Baseline: TF-IDF + 逻辑回归
用于打通完整数据流：读取 -> 预处理 -> 训练 -> 预测 -> 按 sample 格式提交。

数据格式：
    train_set.csv 列: label(0~13), text(空格分隔的匿名数字 token)，'\t' 分隔
    test_a.csv    列: text(同样格式)

运行：
    python src/baseline_tfidf_lr.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# ---------- 路径配置 ----------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE, "data", "train", "train_set.csv")
TEST_PATH = os.path.join(BASE, "data", "test_a", "test_a.csv")
SAMPLE_PATH = os.path.join(BASE, "data", "test_a", "test_a_sample_submit.csv")
SUBMIT_PATH = os.path.join(BASE, "data", "submit", "submit_baseline_tfidf_lr.csv")

LABEL_MAP = {
    '科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6,
    '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13,
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_data():
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    test = pd.read_csv(TEST_PATH, sep="\t")
    print(f"[数据] 训练集 {train.shape[0]} 条, 测试集 {test.shape[0]} 条")
    print(f"[数据] 训练集列: {list(train.columns)}")
    print(f"[数据] 标签分布:\n{train['label'].value_counts().sort_index()}")
    return train, test


def main():
    train, test = load_data()
    X_train, y_train = train["text"].astype(str), train["label"].values
    X_test = test["text"].astype(str)

    # 文本是空格分隔的数字 token，分词器用 split；token 按空格切，用 ngram_range 捕获局部共现
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        ngram_range=(1, 2),
        max_features=50000,
        min_df=5,
        sublinear_tf=True,
    )
    print("[向量化] 开始 TF-IDF 拟合...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[向量化] 特征维度: {X_train_tfidf.shape[1]}")

    clf = LogisticRegression(
        C=4.0,
        solver="liblinear",
        max_iter=200,
        n_jobs=-1,
    )
    print("[训练] 开始逻辑回归训练...")
    clf.fit(X_train_tfidf, y_train)

    # 训练集上的 macro F1（仅参考，非线上分数）
    y_train_pred = clf.predict(X_train_tfidf)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    print(f"[评估] 训练集 macro F1 = {train_f1:.4f}")

    # 预测测试集
    y_pred = clf.predict(X_test_tfidf)
    print("[预测] 完成，类别分布:")
    print(pd.Series(y_pred).value_counts().sort_index())

    # 对齐 sample_submit 格式
    sample = pd.read_csv(SAMPLE_PATH)
    submit = sample.copy()
    submit["label"] = y_pred

    os.makedirs(os.path.dirname(SUBMIT_PATH), exist_ok=True)
    submit.to_csv(SUBMIT_PATH, index=False)
    print(f"[提交] 已写出 {SUBMIT_PATH}, 共 {len(submit)} 行")
    print(f"[提交] 列名: {list(submit.columns)}")
    print("[提示] 上传此文件到天池即可获得线上 macro F1 排名。")


if __name__ == "__main__":
    main()
