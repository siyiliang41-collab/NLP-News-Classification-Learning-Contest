# -*- coding: utf-8 -*-
"""
Baseline: TF-IDF + 线性分类器（SGDClassifier, loss=log_loss 等价逻辑回归）
1. 读数据 -> 2. 5折CV 评估 -> 3. 记录分数 -> 4. 预测 test_a 生成提交文件

选择 SGDClassifier 而非 LogisticRegression(liblinear) 的原因：
    14 类下 liblinear 需训练 14 个 one-vs-rest 二分类器，20w 数据极慢；
    SGD 原生多分类，速度快一个量级，精度与 LR 相当。

运行：
    python src/baseline_tfidf_lr.py
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from config import TRAIN_PATH, TEST_A_PATH, SUBMIT_DIR, SEED, N_FOLDS
from scoreboard import record

SUBMIT_PATH = os.path.join(SUBMIT_DIR, "submit_baseline_tfidf_lr.csv")


def build_vectorizer():
    # token 是空格分隔的纯数字，用默认 C 级 token_pattern 切分（远快于 Python split）
    return TfidfVectorizer(
        token_pattern=r"\d+",     # 只匹配数字 token
        ngram_range=(1, 2),
        max_features=50000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_model():
    # log_loss 即逻辑回归的损失，SGD 求解
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=30,
        random_state=SEED,
        n_jobs=-1,
    )


def main():
    t0 = time.time()
    print("[1/4] 读取数据...")
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    test = pd.read_csv(TEST_A_PATH, sep="\t")
    X = train["text"].astype(str).values
    y = train["label"].values
    print(f"      训练 {len(X)} 条, 测试 {len(test)} 条")

    print("[2/4] 5折交叉验证评估...")
    vectorizer = build_vectorizer()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr = vectorizer.fit_transform(X[tr_idx])
        X_va = vectorizer.transform(X[va_idx])
        clf = build_model()
        clf.fit(X_tr, y[tr_idx])
        y_va_pred = clf.predict(X_va)
        f1 = f1_score(y[va_idx], y_va_pred, average="macro")
        fold_scores.append(f1)
        print(f"      fold {fold}: macro F1 = {f1:.4f}  (用时 {time.time()-t0:.0f}s)")

    cv_f1 = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    print(f"      => CV macro F1 = {cv_f1:.4f} ± {std:.4f}")

    record("tfidf_lr", "baseline", cv_f1, std, "TF-IDF(1-2gram,5w特征) + SGDClassifier(log_loss)")

    print("[3/4] 全量训练并预测 test_a...")
    X_all = vectorizer.fit_transform(X)
    clf = build_model()
    clf.fit(X_all, y)
    y_pred = clf.predict(vectorizer.transform(test["text"].astype(str).values))

    print("[4/4] 生成提交文件...")
    submit = pd.DataFrame({"label": y_pred})
    os.makedirs(SUBMIT_DIR, exist_ok=True)
    submit.to_csv(SUBMIT_PATH, index=False)
    print(f"      已写出 {SUBMIT_PATH}, 共 {len(submit)} 行")
    print(f"[完成] 总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
