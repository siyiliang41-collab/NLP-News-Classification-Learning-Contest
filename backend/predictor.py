# -*- coding: utf-8 -*-
"""
模型推理层：定义统一的 Predictor 接口，并实现 TF-IDF + 逻辑回归 预测器。

设计原则：
- 上层（FastAPI 路由）只依赖 Predictor 接口，不关心具体模型；
- 后续加入 TextCNN / BERT 时，只需新增一个实现类并注册到 REGISTRY，路由代码不动。
"""
import os
import json
import numpy as np
import joblib


class Predictor:
    """所有预测器的统一接口。"""

    name = "base"
    display_name = "Base"

    def predict(self, text: str) -> dict:
        """输入原始文本，返回结构化预测结果。"""
        raise NotImplementedError


class TfidfLrPredictor(Predictor):
    """TF-IDF + 逻辑回归（SGD 求解）预测器。

    加载 train_demo_model.py 保存的 vectorizer.pkl + model.pkl + meta.json。
    """

    name = "tfidf_lr"
    display_name = "TF-IDF + 逻辑回归"

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))
        self.clf = joblib.load(os.path.join(model_dir, "model.pkl"))
        with open(os.path.join(model_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.labels = self.meta["labels"]           # 14 类中文名（按 0~13 顺序）
        self.num_classes = self.meta["num_classes"]

    def predict(self, text: str, topk: int = 3, n_keywords: int = 8) -> dict:
        text = (text or "").strip()
        if not text:
            return {"error": "输入文本为空"}

        x = self.vectorizer.transform([text])
        # loss="log_loss" 的 SGDClassifier 支持 predict_proba（softmax 概率）
        proba = self.clf.predict_proba(x)[0].astype(float)
        label = int(np.argmax(proba))

        # Top-K 候选
        order = np.argsort(-proba)[:topk]
        top3 = [
            {"label": int(i), "name": self.labels[i], "prob": round(float(proba[i]), 4)}
            for i in order
        ]

        result = {
            "label": label,
            "name": self.labels[label],
            "prob": round(float(proba[label]), 4),
            "top3": top3,
            "probs": [round(float(p), 4) for p in proba],   # 14 类完整概率
            "keywords": self._keywords(text, label, n_keywords),
        }
        return result

    def _keywords(self, text: str, label: int, n: int) -> list:
        """可解释性：返回对预测类别贡献最大的 n-gram（token 组合）。

        线性模型对某特征的贡献 = coef[label, feat] * tfidf 值，
        取贡献最大的特征名展示，让前端能高亮「模型为什么这么判」。
        """
        feature_names = self.vectorizer.get_feature_names_out()
        x = self.vectorizer.transform([text])
        coef = self.clf.coef_[label]                      # (n_features,)
        # 稀疏矩阵按非零特征计算贡献
        cx = x.tocoo()
        contribs = []
        for row, col, val in zip(cx.row, cx.col, cx.data):
            contribs.append((feature_names[col], float(coef[col] * val)))
        contribs.sort(key=lambda t: -t[1])
        return [{"term": t, "score": round(s, 4)} for t, s in contribs[:n]]


# 模型注册表：新增模型时在此登记即可
REGISTRY = {
    "tfidf_lr": TfidfLrPredictor,
}


def load_predictor(name: str, models_dir: str) -> Predictor:
    """按名称加载预测器。"""
    if name not in REGISTRY:
        raise ValueError(f"未知模型: {name}，可用: {list(REGISTRY)}")
    model_dir = os.path.join(models_dir, name)
    return REGISTRY[name](model_dir)
