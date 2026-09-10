# -*- coding: utf-8 -*-
"""
训练「演示用」TF-IDF + 逻辑回归模型，并保存可加载的权重，供前端后端推理调用。

与 baseline_tfidf_lr.py 的区别：
1. 切出一块「演示验证池」demo_pool（带真实标签），模型训练时完全不碰它，
   前端「验证演示」模式据此展示「预测 vs 真实标签」并累计正确率。
2. 训练完成后用 joblib 保存 vectorizer + 模型 + 元信息到 models/tfidf_lr/，
   供 backend/predictor.py 反复加载推理（而不是像 baseline 那样只写 submit CSV）。

运行：
    python src/train_demo_model.py
"""
import os
import time
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from config import TRAIN_PATH, MODEL_DIR, LABEL_MAP, ID2LABEL, NUM_CLASSES, SEED

# 演示验证池大小（从训练集切出、模型未见过的带标签样本）
DEMO_POOL_SIZE = 2000
# 模型保存目录
SAVE_DIR = os.path.join(MODEL_DIR, "tfidf_lr")
DEMO_POOL_PATH = os.path.join(os.path.dirname(MODEL_DIR), "data", "demo_pool.csv")


def build_vectorizer():
    # 与 baseline 一致的参数：数字 token、1-2gram、5w 特征、sublinear_tf
    return TfidfVectorizer(
        token_pattern=r"\d+",
        ngram_range=(1, 2),
        max_features=50000,
        min_df=3,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_model():
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        max_iter=30,
        random_state=SEED,
        n_jobs=-1,
    )


def main():
    t0 = time.time()
    print("[1/5] 读取训练数据...")
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    X = train["text"].astype(str).values
    y = train["label"].values
    print(f"      共 {len(X)} 条")

    print(f"[2/5] 切出演示验证池（{DEMO_POOL_SIZE} 条，stratify 保持类别比例）...")
    X_train, X_pool, y_train, y_pool = train_test_split(
        X, y, test_size=DEMO_POOL_SIZE, random_state=SEED, stratify=y,
    )
    # 保存验证池（带真实标签，供前端「验证演示」模式）
    os.makedirs(os.path.dirname(DEMO_POOL_PATH), exist_ok=True)
    pd.DataFrame({"label": y_pool, "text": X_pool}).to_csv(DEMO_POOL_PATH, index=False, sep="\t")
    print(f"      验证池已保存: {DEMO_POOL_PATH}（{len(X_pool)} 条）")
    print(f"      训练剩余 {len(X_train)} 条")

    print("[3/5] 训练 TF-IDF + 逻辑回归...")
    vectorizer = build_vectorizer()
    X_tr = vectorizer.fit_transform(X_train)
    clf = build_model()
    clf.fit(X_tr, y_train)

    # 在验证池上评估（诚实成绩：模型从未见过这些样本）
    y_pool_pred = clf.predict(vectorizer.transform(X_pool))
    pool_f1 = f1_score(y_pool, y_pool_pred, average="macro")
    pool_acc = float((y_pool == y_pool_pred).mean())
    print(f"      验证池 macro F1 = {pool_f1:.4f}, 准确率 = {pool_acc:.4f}")

    print("[4/5] 保存模型权重...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "vectorizer.pkl"))
    joblib.dump(clf, os.path.join(SAVE_DIR, "model.pkl"))
    meta = {
        "name": "tfidf_lr",
        "display_name": "TF-IDF + 逻辑回归",
        "num_classes": NUM_CLASSES,
        "labels": [ID2LABEL[i] for i in range(NUM_CLASSES)],
        "label_map": LABEL_MAP,
        "demo_pool_f1": round(pool_f1, 4),
        "demo_pool_acc": round(pool_acc, 4),
        "demo_pool_size": int(len(X_pool)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(SAVE_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"      已保存到 {SAVE_DIR}/")

    print(f"[5/5] 完成，总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
