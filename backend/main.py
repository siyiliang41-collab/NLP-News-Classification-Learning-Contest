# -*- coding: utf-8 -*-
"""
后端服务：FastAPI 提供 REST API + 托管前端静态页面。

职责：
- 加载演示模型（默认 TF-IDF + 逻辑回归）
- 提供预测 / 随机样本 / 验证演示等接口
- 把 frontend/ 目录作为静态资源托管，浏览器访问 http://localhost:8000 即见页面

运行（根目录执行）：
    python run.py
"""
import os
import sys
import random

# 路径处理：保证无论从哪个目录启动都能 import 到 config 与 predictor
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 项目根目录
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))           # backend 目录

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MODEL_DIR, TEST_A_PATH, NUM_CLASSES, ID2LABEL
from predictor import load_predictor, REGISTRY

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DEMO_POOL_PATH = os.path.join(BASE_DIR, "data", "demo_pool.csv")

app = FastAPI(title="智慧笔迹 · NLP 新闻分类系统", version="1.0.0")

# 允许跨域（本地演示时若用别的端口打开页面也无需额外配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 加载模型与样本数据 ----------
DEFAULT_MODEL = "tfidf_lr"
_predictor = load_predictor(DEFAULT_MODEL, MODEL_DIR)

# 无标签测试集（test_a）用于「随机样本演示」；带标签验证池用于「验证演示」
_test_df = pd.read_csv(TEST_A_PATH, sep="\t")
_pool_df = pd.read_csv(DEMO_POOL_PATH, sep="\t")


# ---------- 请求体模型 ----------
class PredictRequest(BaseModel):
    text: str


# ---------- API ----------
@app.get("/api/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL}


@app.get("/api/meta")
def meta():
    return {
        "labels": [ID2LABEL[i] for i in range(NUM_CLASSES)],
        "num_classes": NUM_CLASSES,
        "models": [
            {"name": name, "display_name": cls.display_name}
            for name, cls in REGISTRY.items()
        ],
        "default_model": DEFAULT_MODEL,
        "demo_pool_size": int(len(_pool_df)),
        "demo_pool_acc": _predictor.meta.get("demo_pool_acc"),
        "demo_pool_f1": _predictor.meta.get("demo_pool_f1"),
    }


@app.post("/api/predict")
def predict(req: PredictRequest):
    result = _predictor.predict(req.text)
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    return result


@app.get("/api/sample")
def sample():
    """从 test_a 随机取一条无标签样本（「随机样本演示」模式）。"""
    idx = random.randint(0, len(_test_df) - 1)
    return {"text": _test_df.iloc[idx]["text"]}


@app.get("/api/demo")
def demo():
    """从验证池随机取一条带标签样本（「验证演示」模式）。

    注意：true_label 一并返回，但由前端负责「先隐藏、预测后再揭晓」。
    """
    idx = random.randint(0, len(_pool_df) - 1)
    row = _pool_df.iloc[idx]
    return {"text": row["text"], "true_label": int(row["label"])}


# ---------- 前端静态页面 ----------
@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
