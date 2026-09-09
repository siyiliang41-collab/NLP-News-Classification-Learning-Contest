# -*- coding: utf-8 -*-
"""
Word2Vec + BiLSTM + Attention 文本分类

作为与 TextCNN（CNN 结构）对比的序列建模（RNN 结构）方案，
补齐报告「算法比较分析」的第四种主流结构。

用法：
    python src/bilstm_attn.py --gpu
    python src/bilstm_attn.py --subset 20000 --epochs 1   # 本地快速验证
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import TRAIN_PATH, TEST_A_PATH, SUBMIT_DIR, SEED, N_FOLDS, NUM_CLASSES
from scoreboard import record
import common

BATCH_SIZE = 256
LR = 1e-3


def build_model(vocab, embed_matrix, device):
    m = common.BiLSTM_Attention(len(vocab), common.EMBED_DIM, hidden_dim=128)
    m.embedding.weight.data.copy_(embed_matrix)
    m.embedding.weight.requires_grad = True
    return m.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="用留出验证集代替5折CV（省时）")
    args = parser.parse_args()

    common.set_seed(SEED)
    device = common.get_device(args.gpu)
    print(f"[设备] {device}")

    t0 = time.time()
    print("[1/6] 读取数据...")
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    test = pd.read_csv(TEST_A_PATH, sep="\t")
    if args.subset:
        train = train.iloc[: args.subset]
    texts = train["text"].astype(str).values
    labels = train["label"].values
    test_texts = test["text"].astype(str).values
    print(f"      训练 {len(texts)} 条, 测试 {len(test_texts)} 条")

    print("[2/6] 构建词表...")
    vocab = common.Vocab(texts)
    print(f"      词表大小: {len(vocab)}")

    print("[3/6] 自训练 Word2Vec 词向量...")
    embed_matrix = common.build_embedding_matrix(texts, vocab, common.EMBED_DIM)

    criterion = nn.CrossEntropyLoss()

    print("[4/6] 评估...")
    if args.holdout:
        tr_texts, va_texts, tr_labels, va_labels = train_test_split(
            texts, labels, test_size=0.1, random_state=SEED, stratify=labels)
        model = build_model(vocab, embed_matrix, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        tr_loader = DataLoader(common.NewsDataset(tr_texts, tr_labels, vocab), batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(common.NewsDataset(va_texts, va_labels, vocab), batch_size=BATCH_SIZE)
        for ep in range(args.epochs):
            loss, acc = common.train_one_epoch(model, tr_loader, optimizer, criterion, device)
            print(f"      ep {ep+1}: loss={loss:.4f} acc={acc:.4f}")
        f1 = common.evaluate(model, va_loader, device)
        print(f"      => val macro F1 = {f1:.4f}")
        record("word2vec_bilstm_attn", "baseline", f1, None,
               f"Word2Vec({common.EMBED_DIM}d)+BiLSTM(128)+Attention, holdout, epochs={args.epochs}")
    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels), 1):
            model = build_model(vocab, embed_matrix, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            tr_ds = common.NewsDataset(texts[tr_idx], labels[tr_idx], vocab)
            va_ds = common.NewsDataset(texts[va_idx], labels[va_idx], vocab)
            tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
            va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)
            for ep in range(args.epochs):
                loss, acc = common.train_one_epoch(model, tr_loader, optimizer, criterion, device)
                print(f"      fold {fold} ep {ep+1}: loss={loss:.4f} acc={acc:.4f}")
            f1 = common.evaluate(model, va_loader, device)
            fold_scores.append(f1)
            print(f"      fold {fold}: macro F1 = {f1:.4f}  (用时 {time.time()-t0:.0f}s)")
        cv_f1 = float(np.mean(fold_scores))
        std = float(np.std(fold_scores))
        print(f"      => CV macro F1 = {cv_f1:.4f} ± {std:.4f}")
        record("word2vec_bilstm_attn", "baseline", cv_f1, std,
               f"Word2Vec({common.EMBED_DIM}d)+BiLSTM(128)+Attention, epochs={args.epochs}")

    print("[5/6] 全量训练...")
    model = build_model(vocab, embed_matrix, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    full_ds = common.NewsDataset(texts, labels, vocab)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    for ep in range(args.epochs):
        loss, acc = common.train_one_epoch(model, full_loader, optimizer, criterion, device)
        print(f"      ep {ep+1}: loss={loss:.4f} acc={acc:.4f}")

    print("[6/6] 预测 test_a 并生成提交...")
    test_ds = common.NewsDataset(test_texts, np.zeros(len(test_texts), dtype=int), vocab)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    y_pred = common.predict(model, test_loader, device)

    os.makedirs(SUBMIT_DIR, exist_ok=True)
    submit = pd.DataFrame({"label": y_pred})
    submit_path = os.path.join(SUBMIT_DIR, "submit_bilstm_attn.csv")
    submit.to_csv(submit_path, index=False)
    print(f"      已写出 {submit_path}, 共 {len(submit)} 行")
    print(f"[完成] 总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
