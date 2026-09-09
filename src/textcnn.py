# -*- coding: utf-8 -*-
"""
Word2Vec + TextCNN 文本分类
1. 用 gensim 在训练集 token 序列上自训练字符级 Word2Vec（匿名化数据无法用现成中文词向量）
2. 构建 TextCNN（多卷积核）模型
3. 5折交叉验证评估 -> 记录分数 -> 预测 test_a 生成提交文件

本地用 CPU 小数据调通；全量训练在 GPU(AutoDL) 上跑：
    python src/textcnn.py --full --gpu
小数据调通：
    python src/textcnn.py --subset 20000 --epochs 1

用法示例：
    python src/textcnn.py --subset 20000 --epochs 1          # CPU 快速验证
    python src/textcnn.py --full --gpu                       # 全量 GPU 训练
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import (
    TRAIN_PATH, TEST_A_PATH, SUBMIT_DIR, MODEL_DIR,
    SEED, N_FOLDS, NUM_CLASSES, ID2LABEL,
)
from scoreboard import record


# ---------- 超参 ----------
MAX_SEQ_LEN = 600        # 截断长度
EMBED_DIM = 128          # 词向量维度（gensim Word2Vec 维度，需一致）
NUM_FILTERS = 100        # 每种卷积核个数
FILTER_SIZES = [2, 3, 4]  # 卷积核大小（字符级 n-gram 窗口）
DROPOUT = 0.5
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 5
VOCAB_MIN_COUNT = 1      # 词表最小词频（匿名 token 频次差异大，保留全部）


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------- 词汇表 ----------
class Vocab:
    def __init__(self, texts, min_count=VOCAB_MIN_COUNT, max_size=None):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        # 按频次排序，保留 top max_size
        words = [w for w, c in counter.most_common(max_size) if c >= min_count]
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        for w in words:
            self.stoi[w] = len(self.stoi)
        self.itos = {v: k for k, v in self.stoi.items()}

    def encode(self, text, max_len=MAX_SEQ_LEN):
        ids = [self.stoi.get(w, 1) for w in text.split()[:max_len]]
        # padding
        ids = ids + [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.stoi)


class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(self.vocab.encode(self.texts[idx]), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------- 模型 ----------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, filter_sizes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))            # (batch, num_filters, seq_len-fs+1, 1)
            c = c.squeeze(3)                 # (batch, num_filters, L)
            pooled.append(F.max_pool1d(c, c.size(2)).squeeze(2))  # (batch, num_filters)
        out = torch.cat(pooled, dim=1)       # (batch, num_filters * len(filter_sizes))
        out = self.dropout(out)
        return self.fc(out)


# ---------- 训练 ----------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
    from sklearn.metrics import f1_score
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return f1_score(trues, preds, average="macro")


def load_word2vec_embeddings(vocab, dim=EMBED_DIM):
    """
    在训练集 token 序列上自训练 Word2Vec，返回嵌入矩阵 (vocab_size, dim)。
    匿名化数据没有现成词向量，只能自训练。
    """
    from gensim.models import Word2Vec
    texts = [t.split()[:MAX_SEQ_LEN] for t in vocab_texts_holder]
    w2v = Word2Vec(
        sentences=texts, vector_size=dim, window=5, min_count=VOCAB_MIN_COUNT,
        workers=4, sg=1, epochs=5, seed=SEED,
    )
    emb = np.random.normal(0, 0.1, (len(vocab), dim))
    emb[0] = 0.0  # PAD
    hit = 0
    for w, i in vocab.stoi.items():
        if w in w2v.wv:
            emb[i] = w2v.wv[w]
            hit += 1
    print(f"[Word2Vec] 命中 {hit}/{len(vocab)} 个 token 的向量")
    return torch.tensor(emb, dtype=torch.float32)


# 用于 Word2Vec 训练的原始文本（由 main 设置）
vocab_texts_holder = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, help="只取前 N 条用于快速调通")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--gpu", action="store_true", help="使用 GPU")
    parser.add_argument("--no-cv", action="store_true", help="跳过交叉验证，直接全量训练预测")
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device(args.gpu)
    print(f"[设备] {device}")

    t0 = time.time()
    print("[1/6] 读取数据...")
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    test = pd.read_csv(TEST_A_PATH, sep="\t")
    if args.subset:
        train = train.iloc[: args.subset]
        print(f"      (subset 模式，只取前 {args.subset} 条)")
    texts = train["text"].astype(str).values
    labels = train["label"].values
    test_texts = test["text"].astype(str).values
    print(f"      训练 {len(texts)} 条, 测试 {len(test_texts)} 条")

    print("[2/6] 构建词表...")
    vocab = Vocab(texts)
    print(f"      词表大小: {len(vocab)}")

    print("[3/6] 自训练 Word2Vec 词向量...")
    global vocab_texts_holder
    vocab_texts_holder = texts
    embed_matrix = load_word2vec_embeddings(vocab, EMBED_DIM)

    def build_model():
        m = TextCNN(len(vocab), EMBED_DIM, NUM_CLASSES, NUM_FILTERS, FILTER_SIZES, DROPOUT)
        m.embedding.weight.data.copy_(embed_matrix)
        m.embedding.weight.requires_grad = True  # 允许微调
        return m.to(device)

    criterion = nn.CrossEntropyLoss()

    if not args.no_cv:
        print("[4/6] 5折交叉验证评估...")
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels), 1):
            model = build_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            tr_ds = NewsDataset(texts[tr_idx], labels[tr_idx], vocab)
            va_ds = NewsDataset(texts[va_idx], labels[va_idx], vocab)
            tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)
            for ep in range(args.epochs):
                loss, acc = train_one_epoch(model, tr_loader, optimizer, criterion, device)
                print(f"      fold {fold} ep {ep+1}: loss={loss:.4f} acc={acc:.4f}")
            f1 = evaluate(model, va_loader, device)
            fold_scores.append(f1)
            print(f"      fold {fold}: macro F1 = {f1:.4f}  (用时 {time.time()-t0:.0f}s)")
        cv_f1 = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        print(f"      => CV macro F1 = {cv_f1:.4f} ± {std:.4f}")
        record("word2vec_textcnn", "baseline", cv_f1, std,
               f"Word2Vec({EMBED_DIM}d)+TextCNN(filters={FILTER_SIZES}), epochs={args.epochs}")

    print("[5/6] 全量训练...")
    model = build_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    full_ds = NewsDataset(texts, labels, vocab)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(args.epochs):
        loss, acc = train_one_epoch(model, full_loader, optimizer, criterion, device)
        print(f"      ep {ep+1}: loss={loss:.4f} acc={acc:.4f}")

    print("[6/6] 预测 test_a 并生成提交...")
    model.eval()
    test_ds = NewsDataset(test_texts, np.zeros(len(test_texts), dtype=int), vocab)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)

    os.makedirs(SUBMIT_DIR, exist_ok=True)
    submit = pd.DataFrame({"label": y_pred})
    submit_path = os.path.join(SUBMIT_DIR, "submit_word2vec_textcnn.csv")
    submit.to_csv(submit_path, index=False)
    print(f"      已写出 {submit_path}, 共 {len(submit)} 行")
    print(f"[完成] 总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
