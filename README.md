# 智慧笔迹 · NLP 新闻分类学习赛

> 26 秋小学期课设（源自天池大赛「零基础入门 NLP——新闻文本分类」）
>
> 仓库：https://github.com/siyiliang41-collab/NLP-News-Classification-Learning-Contest

**14 类新闻文本多分类**任务，数据为**字符级匿名化**，评测指标为 **macro F1**。项目走完「传统机器学习基线 → 深度学习 → 预训练模型 → 多模型融合」的完整技术路线，最终把分数从 **0.8463** 拉升到 **0.9332**，并交付了一套可复用的工程结构（统一配置、分数台账、可视化、答辩演示界面）。

---

## 赛题速览

| 项目 | 说明 |
|------|------|
| 任务 | 输入新闻文本 → 输出 14 类之一 |
| 类别 | 科技 / 股票 / 体育 / 娱乐 / 时政 / 社会 / 教育 / 财经 / 家居 / 游戏 / 房产 / 时尚 / 彩票 / 星座 |
| 训练集 | 20w 条（`train_set.csv`，列 `label` + `text`，`\t` 分隔） |
| 测试集 | A 集 5w 条（`test_a.csv`）；B 集 5w 条（后期放出，本课设未使用） |
| 评测 | macro F1（各类别 F1 均值） |
| 数据特点 | 文本按**字符级匿名化**，空格分隔的数字 token，分词与现成预训练词向量全部失效 |

**标签映射（官方固定，勿改动）：**

```python
label_map = {
    '科技': 0, '股票': 1, '体育': 2, '娱乐': 3, '时政': 4, '社会': 5, '教育': 6,
    '财经': 7, '家居': 8, '游戏': 9, '房产': 10, '时尚': 11, '彩票': 12, '星座': 13
}
```

---

## 成绩阶梯（5 折 CV 下的 macro F1）

| 阶段 | 模型 | macro F1 | 提升 | 备注 |
|------|------|----------|------|------|
| ① 基线 | TF-IDF + LR | **0.8463** ± 0.0038 | — | 5 折 CV，SGDClassifier(log_loss) |
| ② 深度·CNN | Word2Vec + TextCNN | **0.9183** ± 0.0024 | **+7.2** | 5 折 CV，skip-gram，filters=[2,3,4] |
| ③ 深度·RNN | Word2Vec + BiLSTM + Attn | **0.9241** | +0.6 | holdout，BiLSTM(128)+Attention |
| ④ 预训练 | BERT 微调 | **0.9240** | ≈持平 | bert-base-chinese，holdout |
| ⑤ 融合 | 硬投票（BERT + TextCNN + TFIDF-LR） | **0.9332** | **+0.9** | 最终最高分，总提升 +8.7 |

> 分数统一登记在 [`docs/scores.csv`](docs/scores.csv) 台账中，可视化见 [`src/plot_scores.py`](src/plot_scores.py)。

---

## 技术路线（逐级上分）

```
TF-IDF + LR (baseline) → Word2Vec + TextCNN
  → BiLSTM + Attention → BERT 微调 → 多模型硬投票融合
```

选这条路的原因：**数据是字符级匿名的**，中文分词和现成的预训练词向量全都用不了，所以先在训练集上**自训练字符级 Word2Vec**，再让 CNN 抓局部 n-gram 特征、RNN 建模长程依赖，最后靠「不同结构模型犯错不同」的集成互补冲分。

---

## 核心洞察（三个可写进报告的结论）

1. **匿名化 → 分词/预训练词向量全失效 → 自训练字符级 Word2Vec**：匿名化只是「把词典换了」，token 的共现关系、n-gram 模式、出现频率等统计结构被完整保留，所以自训练词向量这条路仍然成立。
2. **BERT 没有赢过 BiLSTM（0.9240 vs 0.9241）**：匿名化让预训练语义基本失效，BERT 退化为「更贵的从零学习」，说明项目不是盲目堆模型。
3. **损失函数消融发现，数据瓶颈是「类间混淆」而非「类别不均衡」**：加权交叉熵 / Focal Loss 反而降分（CE 0.9042 → 加权 0.8750 / Focal 0.8886），真正难分的是社会、时政、房产这些大类的互相混淆。

---

## EDA 关键发现

| 发现 | 数据 | 结论与动作 |
|------|------|-----------|
| 类别严重不均衡 | 科技 38,918 条 vs 星座 908 条，约 **43 : 1** | macro F1 下小类同等重要 → 交叉验证用 `StratifiedKFold` |
| 文本长度长尾 | 均值 907、中位 676、最长 12,211，≤600 约占 44.5% | 截断 `MAX_SEQ_LEN = 600`，保留标题/导语信息 |
| 超高频 token | `#3750` 出现 748 万次 | 疑为匿名化前的标点或高频停用字，保留由模型自行学习 |

详见 [`docs/EDA分析报告.md`](docs/EDA分析报告.md)。

---

## 目录结构

```
.
├── backend/          # FastAPI 后端：REST API + 托管前端静态页
│   ├── main.py       #   路由、样本/预测/验证接口
│   └── predictor.py  #   统一 Predictor 接口 + TF-IDF/LR 预测器
├── frontend/         # 前端界面（HTML + JS + CSS，ECharts 概率图）
├── src/              # 源码
│   ├── baseline_tfidf_lr.py   # ① TF-IDF + LR 基线
│   ├── textcnn.py             # ② Word2Vec + TextCNN
│   ├── textcnn_loss.py        #    损失函数消融（CE/加权/Focal）
│   ├── bilstm_attn.py         # ③ Word2Vec + BiLSTM + Attention
│   ├── bert.py                # ④ BERT 微调
│   ├── ensemble.py            # ⑤ 硬投票/加权投票融合
│   ├── eda.py                 # 数据探索分析
│   ├── plot_scores.py         # 分数阶梯可视化
│   ├── plot_loss_ablation.py  # 消融实验可视化
│   ├── scoreboard.py          # 分数台账（写 docs/scores.csv）
│   ├── train_demo_model.py    # 训练「演示用」模型 + 切验证池
│   ├── common.py              # 公共组件（词表/数据集/模型/损失/训练循环）
│   └── config.py              # 统一配置（路径/类别映射/超参）
├── data/
│   ├── train/        # 训练集原始数据
│   ├── test_a/       # 测试集 A 原始数据
│   ├── demo_pool.csv # 演示验证池（2000 条带标签，模型从未见过）
│   └── submit/       # 提交文件（.csv）
├── models/
│   └── tfidf_lr/     # 演示模型权重（vectorizer + model + meta）
├── docs/             # 报告、分数台账等文档
├── notebooks/        # 探索性分析 notebook
├── output/figs/      # 各类图表（进报告用）
├── run.py            # 一键启动前后端服务
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

将数据放入 `data/train/train_set.csv` 与 `data/test_a/test_a.csv`（`config.py` 支持本地/云端多路径自动探测，免改路径）。

### 3. 复现实验（按技术路线逐级跑）

```bash
python src/baseline_tfidf_lr.py    # ① 基线 0.8463
python src/textcnn.py              # ② TextCNN 0.9183
python src/bilstm_attn.py          # ③ BiLSTM+Attn 0.9241
python src/bert.py                 # ④ BERT 0.9240（需 GPU）
python src/ensemble.py --files ... # ⑤ 硬投票融合 0.9332
```

### 4. 启动答辩演示界面

```bash
# 先训练演示模型（生成 models/tfidf_lr/ 权重 + data/demo_pool.csv 验证池）
python src/train_demo_model.py

# 一键启动前后端
python run.py
```

浏览器打开 **http://localhost:8000**（API 文档 http://localhost:8000/docs）。

---

## Web 演示系统

基于 **FastAPI + 静态前端（ECharts）** 的交互式演示界面，提供三种模式：

| 模式 | 说明 |
|------|------|
| 随机样本演示 | 从测试集（无标签）随机抽取一条真实新闻，观察模型判断 |
| 验证演示 | 从「模型从未见过的验证池」（2000 条）抽取带标签样本，预测后揭晓对错并累计正确率 |
| 手动输入 | 粘贴「空格分隔的数字 token」文本进行预测 |

每次预测展示 **Top-3 候选 + 14 类概率条形图 + 关键 n-gram 特征**（可解释性）。页脚显示验证池规模、离线 macro F1（0.7984）、准确率（89.25%）。

---

## 团队分工

| 成员 | 负责方向 |
|------|---------|
| 梁思怡 | BERT、BiLSTM+Attention（0.9240 / 0.9241） |
| 李思霖 | 数据与 EDA（发现类别不均衡、长尾长度两个关键问题） |
| 曹煜权 | TextCNN、损失函数消融（TextCNN 0.9183） |
| 胡子琦 | baseline、数据链路、模型融合（baseline 0.8463 / 融合 0.9332） |
| 黄子涵 | 可视化、前端界面、测试 |

协作方式：git 统一管理代码，实验分数统一登记在 `docs/scores.csv` 台账。

---

## 环境

- Python 3.10 + PyTorch + HuggingFace transformers
- 本地 CPU 跑 baseline 与快速验证；深度模型在阿里云 PAI-DSW 的 T4 16G GPU 上训练

## 参考文档

- [`docs/课设详解.md`](docs/课设详解.md) —— 各级算法原理与实验的完整详解
- [`docs/EDA分析报告.md`](docs/EDA分析报告.md) —— 数据探索分析报告
- [`docs/scores.csv`](docs/scores.csv) —— 所有实验分数台账
