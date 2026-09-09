# -*- coding: utf-8 -*-
"""
类别不均衡优化实验：在 TextCNN 上对比三种损失函数
  1. ce       普通交叉熵（baseline）
  2. weighted 类别权重交叉熵（balanced 权重）
  3. focal    Focal Loss（聚焦难样本）

目的是呼应 EDA 发现的「类别不均衡 43 倍」难点，验证优化手段对 macro F1 的增益。

用法：
    python src/textcnn_loss.py --gpu --loss ce        # 单跑一种
    python src/textcnn_loss.py --gpu --loss all       # 三种全跑（报告对比用）
    python src/textcnn_loss.py --subset 20000 --loss all --epochs 1   # 本地快速验证
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

from config import TRAIN_PATH, TEST_A_PATH, SUBMIT_DIR, SEED, N_FOLDS, NUM_CLASSES, ID2LABEL
from scoreboard import record
import common


def build_model(vocab, embed_matrix, device):
    m = common.TextCNN(len(vocab), common.EMBED_DIM, NUM_CLASSES)
    m.embedding.weight.data.copy_(embed_matrix)
    m.embedding.weight.requires_grad = True
    return m.to(device)


def build_criterion(loss_name, labels, device):
    if loss_name == "ce":
        return nn.CrossEntropyLoss(), "普通交叉熵 CE"
    if loss_name == "weighted":
        w = common.compute_class_weights(labels, NUM_CLASSES).to(device)
        return nn.CrossEntropyLoss(weight=w), "类别权重交叉熵 (balanced)"
    if loss_name == "focal":
        w = common.compute_class_weights(labels, NUM_CLASSES).to(device)
        return common.FocalLoss(gamma=2.0, alpha=w), "Focal Loss (gamma=2, 带权重)"
    raise ValueError(loss_name)


def run_cv(texts, labels, vocab, embed_matrix, loss_name, epochs, device):
    """5折CV，返回 (cv_f1, std)。"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels), 1):
        model = build_model(vocab, embed_matrix, device)
        criterion, _ = build_criterion(loss_name, labels[tr_idx], device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        tr_ds = common.NewsDataset(texts[tr_idx], labels[tr_idx], vocab)
        va_ds = common.NewsDataset(texts[va_idx], labels[va_idx], vocab)
        tr_loader = DataLoader(tr_ds, batch_size=256, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=256)
        for ep in range(epochs):
            common.train_one_epoch(model, tr_loader, optimizer, criterion, device)
        f1 = common.evaluate(model, va_loader, device)
        fold_scores.append(f1)
        print(f"      fold {fold}: macro F1 = {f1:.4f}")
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def run_holdout(texts, labels, vocab, embed_matrix, loss_name, epochs, device):
    """留出验证集（省时）：90%训练/10%验证，单次训练，返回 (val_f1, per_class_f1)。"""
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels)
    model = build_model(vocab, embed_matrix, device)
    criterion, _ = build_criterion(loss_name, tr_labels, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_loader = DataLoader(common.NewsDataset(tr_texts, tr_labels, vocab), batch_size=256, shuffle=True)
    va_loader = DataLoader(common.NewsDataset(va_texts, va_labels, vocab), batch_size=256)
    for ep in range(epochs):
        loss, acc = common.train_one_epoch(model, tr_loader, optimizer, criterion, device)
        print(f"      ep {ep+1}: loss={loss:.4f} acc={acc:.4f}")
    f1 = common.evaluate(model, va_loader, device)
    per_class = common.evaluate_per_class(model, va_loader, device, ID2LABEL)
    print(f"      val macro F1 = {f1:.4f}")
    print(f"      各类别 F1: {per_class}")
    return float(f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", default="all", choices=["ce", "weighted", "focal", "all"])
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="用留出验证集代替5折CV（省时）")
    args = parser.parse_args()

    common.set_seed(SEED)
    device = common.get_device(args.gpu)
    print(f"[设备] {device}")

    t0 = time.time()
    train = pd.read_csv(TRAIN_PATH, sep="\t")
    test = pd.read_csv(TEST_A_PATH, sep="\t")
    if args.subset:
        train = train.iloc[: args.subset]
    texts = train["text"].astype(str).values
    labels = train["label"].values
    print(f"[数据] 训练 {len(texts)} 条")

    vocab = common.Vocab(texts)
    print(f"[词表] 大小 {len(vocab)}")
    embed_matrix = common.build_embedding_matrix(texts, vocab, common.EMBED_DIM)

    loss_list = ["ce", "weighted", "focal"] if args.loss == "all" else [args.loss]
    for ln in loss_list:
        print(f"\n[实验] 损失函数 = {ln}")
        if args.holdout:
            f1 = run_holdout(texts, labels, vocab, embed_matrix, ln, args.epochs, device)
            record("textcnn", f"loss_{ln}", f1, None,
                   f"TextCNN + {build_criterion(ln, labels, device)[1]}, holdout, epochs={args.epochs}")
        else:
            cv_f1, std = run_cv(texts, labels, vocab, embed_matrix, ln, args.epochs, device)
            print(f"      => CV macro F1 = {cv_f1:.4f} ± {std:.4f}")
            record("textcnn", f"loss_{ln}", cv_f1, std,
                   f"TextCNN + {build_criterion(ln, labels, device)[1]}, epochs={args.epochs}")

    print(f"\n[完成] 总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
