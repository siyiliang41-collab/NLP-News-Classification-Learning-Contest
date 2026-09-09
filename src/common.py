# -*- coding: utf-8 -*-
"""
公共组件：词表、数据集、模型、训练/评估循环、Word2Vec 加载、损失函数。
供 textcnn_loss.py（类别不均衡实验）与 bilstm_attn.py 复用。

运行脚本时（如 `python src/xxx.py`），脚本所在目录 src/ 会被加入 sys.path，
因此这里的 `from config import ...` 与 `import common` 均能正常解析。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import Counter

from config import SEED, NUM_CLASSES

# ---------- 超参 ----------
MAX_SEQ_LEN = 600        # 文本截断长度
EMBED_DIM = 128          # 词向量维度
VOCAB_MIN_COUNT = 1


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu):
    if gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------- 词表 ----------
class Vocab:
    def __init__(self, texts, min_count=VOCAB_MIN_COUNT, max_size=None):
        counter = Counter()
        for t in texts:
            counter.update(t.split())
        words = [w for w, c in counter.most_common(max_size) if c >= min_count]
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        for w in words:
            self.stoi[w] = len(self.stoi)
        self.itos = {v: k for k, v in self.stoi.items()}

    def encode(self, text, max_len=MAX_SEQ_LEN):
        ids = [self.stoi.get(w, 1) for w in text.split()[:max_len]]
        return ids + [0] * (max_len - len(ids))

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
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES,
                 num_filters=100, filter_sizes=(2, 3, 4), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).unsqueeze(1)               # (batch, 1, seq, embed)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))                          # (batch, nf, seq-fs+1, 1)
            c = c.squeeze(3)
            pooled.append(F.max_pool1d(c, c.size(2)).squeeze(2))
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        return self.fc(out)


class BiLSTM_Attention(nn.Module):
    """双向 LSTM + 注意力：注意力层自动学习每个时间步对分类的重要程度。"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=128,
                 num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                            # (batch, seq, embed)
        out, _ = self.lstm(emb)                            # (batch, seq, 2*hidden)
        scores = torch.softmax(self.attn(out), dim=1)      # (batch, seq, 1) 注意力权重
        context = torch.sum(out * scores, dim=1)           # 加权求和 (batch, 2*hidden)
        context = self.dropout(context)
        return self.fc(context)


# ---------- 损失函数 ----------
class FocalLoss(nn.Module):
    """Focal Loss：让模型聚焦难分类样本，缓解类别不均衡。
    gamma 越大越聚焦难样本；alpha 为类别权重（可选）。"""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor (num_classes,) 或 None

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")  # (batch,)
        pt = torch.exp(-ce)                                     # 正确类别的概率
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            focal = self.alpha[target] * focal
        return focal.mean()


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """balanced 类别权重：样本越少的类权重越大，再归一化到均值 1。"""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------- Word2Vec ----------
def train_word2vec(texts, dim=EMBED_DIM, sg=0, epochs=3):
    """在训练集 token 序列上自训练字符级 Word2Vec，返回 gensim 模型。
    sg=0(CBOW) 速度快、分类场景够用；sg=1(skip-gram) 更慢但更精。"""
    from gensim.models import Word2Vec
    sentences = [t.split()[:MAX_SEQ_LEN] for t in texts]
    return Word2Vec(
        sentences=sentences, vector_size=dim, window=5,
        min_count=VOCAB_MIN_COUNT, workers=4, sg=sg, epochs=epochs, seed=SEED,
    )


def build_embedding_matrix(texts, vocab, dim=EMBED_DIM):
    """训练 Word2Vec 并映射到词表，返回嵌入矩阵 (vocab_size, dim)。"""
    w2v = train_word2vec(texts, dim)
    emb = np.random.normal(0, 0.1, (len(vocab), dim))
    emb[0] = 0.0  # PAD
    hit = 0
    for w, i in vocab.stoi.items():
        if w in w2v.wv:
            emb[i] = w2v.wv[w]
            hit += 1
    print(f"[Word2Vec] 命中 {hit}/{len(vocab)} 个 token 的向量")
    return torch.tensor(emb, dtype=torch.float32)


# ---------- 训练 / 评估 ----------
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
    """返回 macro F1。"""
    from sklearn.metrics import f1_score
    preds, trues = _collect(model, loader, device)
    return f1_score(trues, preds, average="macro")


def evaluate_per_class(model, loader, device, id2label):
    """返回各类别 F1 字典，用于类别不均衡分析。"""
    from sklearn.metrics import f1_score
    preds, trues = _collect(model, loader, device)
    f1 = f1_score(trues, preds, average=None, labels=range(len(id2label)))
    return {id2label[i]: round(float(v), 4) for i, v in enumerate(f1)}


def _collect(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)
