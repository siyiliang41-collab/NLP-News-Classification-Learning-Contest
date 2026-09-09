# -NLP-

26秋小学期课设（from 天池大赛）—— 智慧笔迹：NLP 新闻分类学习赛。

14 类新闻文本分类，字符级匿名化数据，评测指标为 **macro F1**。

## 赛题速览

| 项目 | 说明 |
|------|------|
| 任务 | 输入新闻文本 → 输出 14 类之一 |
| 类别 | 财经 / 彩票 / 房产 / 股票 / 家居 / 教育 / 科技 / 社会 / 时尚 / 时政 / 体育 / 星座 / 游戏 / 娱乐 |
| 训练集 | 20w 条（`train_set.csv`，列 `label` + `text`，`\t` 分隔） |
| 测试集 | A 集 5w 条（`test_a.csv`）；B 集 5w 条（后期放出） |
| 评测 | macro F1（各类别 F1 均值） |
| 数据特点 | 文本按**字符级匿名化**，空格分隔的数字 token |

## 标签映射（官方固定，勿改动）

```python
label_map = {
    '科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6,
    '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13
}
```

## 目录结构

```
.
├── data/
│   ├── train/        # 训练集原始数据
│   ├── test_a/       # 测试集 A 原始数据
│   └── submit/       # 提交文件（.csv）
├── src/              # 源码
├── models/           # 训练好的模型权重
├── notebooks/        # 探索性分析 notebook
├── output/figs/      # 各类图表（进报告用）
└── docs/             # 规划、报告等文档
```

## 技术路线（逐级上分）

```
TF-IDF + LR (baseline) → Word2Vec + TextCNN
  → BiLSTM + Attention → BERT 微调 → 多模型融合
```

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 数据放入 `data/train/train_set.csv` 与 `data/test_a/test_a.csv`
3. 跑 baseline：`python src/baseline_tfidf_lr.py`

## 环境

- Python 3.10 + PyTorch + HuggingFace transformers
- 本地 CPU 跑基线 / AutoDL 租 GPU 跑 BERT

## 团队分工

见 `docs/` 下的规划文档。
