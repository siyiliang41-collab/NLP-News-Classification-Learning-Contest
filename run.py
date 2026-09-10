# -*- coding: utf-8 -*-
"""
一键启动前后端服务。

前提：
    1. 已运行 `python src/train_demo_model.py` 生成模型权重（models/tfidf_lr/）
    2. 依赖已安装：pip install -r requirements.txt（含 fastapi uvicorn）

运行：
    python run.py
然后浏览器打开 http://localhost:8000
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

import uvicorn
from main import app

if __name__ == "__main__":
    print("=" * 56)
    print("  智慧笔迹 · NLP 新闻分类系统")
    print("  访问地址: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print("=" * 56)
    uvicorn.run(app, host="0.0.0.0", port=8000)
