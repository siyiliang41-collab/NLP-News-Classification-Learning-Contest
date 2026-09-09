# -*- coding: utf-8 -*-
"""
BERT 微调文本分类（预训练模型阶段）

关键说明：
- 文本是「字符级匿名化」的数字 token，用空格连接后喂给 bert-base-chinese；
  BERT 的预训练语义帮不上太多，主要靠微调学习 token 共现模式。
- 为控制成本，默认用「留出验证集」而非 5 折 CV（5折 = 5倍训练时间）。
- 支持 checkpoint 续跑（防竞价/意外中断，普通实例也用得上）。

省钱默认参数（可覆盖）：
    python src/bert.py --subset 50000 --max_len 256 --epochs 3 --gpu
全量（贵，慎用）：
    python src/bert.py --full --max_len 512 --epochs 3 --gpu
本地快速调通：
    python src/bert.py --subset 2000 --max_len 128 --epochs 1

模型下载（国内）：
    export HF_ENDPOINT=https://hf-mirror.com
"""
import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

from config import TRAIN_PATH, TEST_A_PATH, SUBMIT_DIR, MODEL_DIR, SEED, NUM_CLASSES
from scoreboard import record

PRETRAINED = "bert-base-chinese"
BATCH_SIZE = 32
LR = 2e-5
VAL_RATIO = 0.1


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for input_ids, mask, y in loader:
        input_ids, mask, y = input_ids.to(device), mask.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(input_ids, attention_mask=mask, labels=y)
        loss = out.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for input_ids, mask, y in loader:
            input_ids, mask = input_ids.to(device), mask.to(device)
            out = model(input_ids, attention_mask=mask)
            preds.append(out.logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
    return f1_score(np.concatenate(trues), np.concatenate(preds), average="macro")


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for input_ids, mask, _ in loader:
            input_ids, mask = input_ids.to(device), mask.to(device)
            out = model(input_ids, attention_mask=mask)
            preds.append(out.logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=50000, help="训练子集大小，0=全量")
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu", action="store_true")
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
    texts = train["text"].astype(str).values
    labels = train["label"].values
    test_texts = test["text"].astype(str).values
    print(f"      训练 {len(texts)} 条, 测试 {len(test_texts)} 条")

    print("[2/6] 加载 tokenizer 与模型...")
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED)
    model = BertForSequenceClassification.from_pretrained(PRETRAINED, num_labels=NUM_CLASSES).to(device)

    # 划分训练/验证
    tr_texts, va_texts, tr_labels, va_labels = train_test_split(
        texts, labels, test_size=VAL_RATIO, random_state=SEED, stratify=labels)
    tr_ds = NewsDataset(tr_texts, tr_labels, tokenizer, args.max_len)
    va_ds = NewsDataset(va_texts, va_labels, tokenizer, args.max_len)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = len(tr_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    ckpt_path = os.path.join(MODEL_DIR, "bert_checkpoint.pt")
    os.makedirs(MODEL_DIR, exist_ok=True)
    start_epoch = 0
    if os.path.exists(ckpt_path):
        start_epoch = load_checkpoint(model, optimizer, ckpt_path, device) + 1
        print(f"      [续跑] 从 epoch {start_epoch} 继续")

    print("[3/6] 训练...")
    best_f1 = 0.0
    for ep in range(start_epoch, args.epochs):
        loss, acc = train_one_epoch(model, tr_loader, optimizer, scheduler, device)
        f1 = evaluate(model, va_loader, device)
        print(f"      ep {ep+1}: loss={loss:.4f} acc={acc:.4f} val_macroF1={f1:.4f} (用时 {time.time()-t0:.0f}s)")
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(model, optimizer, ep, ckpt_path)

    print(f"[4/6] 最佳验证 macro F1 = {best_f1:.4f}")
    record("bert", f"base-chinese_maxlen{args.max_len}_sub{args.subset}_ep{args.epochs}",
           best_f1, None, f"bert-base-chinese 微调, val set, {len(texts)} 训练样本")

    print("[5/6] 加载最佳模型并预测 test_a...")
    if os.path.exists(ckpt_path):
        load_checkpoint(model, optimizer, ckpt_path, device)

    test_ds = NewsDataset(test_texts, np.zeros(len(test_texts), dtype=int), tokenizer, args.max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    y_pred = predict(model, test_loader, device)

    print("[6/6] 生成提交文件...")
    submit = pd.DataFrame({"label": y_pred})
    submit_path = os.path.join(SUBMIT_DIR, "submit_bert.csv")
    submit.to_csv(submit_path, index=False)
    print(f"      已写出 {submit_path}, 共 {len(submit)} 行")
    print(f"[完成] 总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
