# -*- coding: utf-8 -*-
"""
系统交互界面（Gradio）：答辩现场演示用。
1. 输入新闻文本 → 预测类别 + 各类别概率条形图
2. 从测试集随机取样本演示（展示真实效果）

注意：赛题文本是字符级匿名化（空格分隔数字），普通中文输入无法直接预测，
因此演示用「随机取测试样本」为主，输入框为辅（输入匿名 token 文本也可预测）。

用法：
    python src/app.py --model bert          # 用 BERT 模型
    python src/app.py --model textcnn       # 用 TextCNN 模型（需已训练并保存权重）
"""
import os
import random
import argparse
import numpy as np
import pandas as pd
import gradio as gr

import torch
from config import TRAIN_PATH, TEST_A_PATH, MODEL_DIR, NUM_CLASSES, ID2LABEL
import common

# 分类标签中文名（按 0~13 顺序）
LABEL_NAMES = [ID2LABEL[i] for i in range(NUM_CLASSES)]


def load_bert_model(device):
    """加载 bert-base-chinese 微调模型（需已运行 bert.py 并保存权重）。"""
    from transformers import BertTokenizer, BertForSequenceClassification
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=NUM_CLASSES).to(device)
    ckpt = os.path.join(MODEL_DIR, "bert_checkpoint.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print("[加载] BERT 权重已加载")
    else:
        print("[警告] 未找到 BERT 权重，将使用未微调的模型（演示效果差）")
    model.eval()
    return model, tokenizer


def load_textcnn_model(device):
    """加载 TextCNN（需词表 + 权重，此处简化为随机初始化演示）。"""
    # 读取少量数据建词表
    train = pd.read_csv(TRAIN_PATH, sep="\t", nrows=5000)
    vocab = common.Vocab(train["text"].astype(str).values)
    model = common.TextCNN(len(vocab), common.EMBED_DIM, NUM_CLASSES).to(device)
    model.eval()
    return model, vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert", choices=["bert", "textcnn"])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = common.get_device(args.gpu)

    if args.model == "bert":
        model, tokenizer = load_bert_model(device)

        def predict_fn(text):
            enc = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = int(np.argmax(probs))
            return LABEL_NAMES[top_idx], {LABEL_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    else:  # textcnn
        model, vocab = load_textcnn_model(device)

        def predict_fn(text):
            x = torch.tensor(vocab.encode(text), dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            top_idx = int(np.argmax(probs))
            return LABEL_NAMES[top_idx], {LABEL_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

    # 从测试集随机取样演示
    test_df = pd.read_csv(TEST_A_PATH, sep="\t")

    def random_sample():
        idx = random.randint(0, len(test_df) - 1)
        text = test_df.iloc[idx]["text"]
        label, probs = predict_fn(text)
        return text[:200] + "...", label

    with gr.Blocks(title="智慧笔迹：NLP 新闻分类系统") as demo:
        gr.Markdown("# 智慧笔迹：NLP 新闻分类系统\n"
                    "14 类新闻文本分类（财经/彩票/房产/股票/家居/教育/科技/社会/时尚/时政/体育/星座/游戏/娱乐）")
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(label="输入新闻文本（匿名 token）", lines=5,
                                        placeholder="粘贴空格分隔的数字 token 文本...")
                btn = gr.Button("预测", variant="primary")
                sample_btn = gr.Button("随机取一条测试样本")
            with gr.Column():
                label_output = gr.Label(label="预测类别")
                prob_plot = gr.BarPlot(label="各类别概率", x="label", y="prob")
        btn.click(predict_fn, inputs=text_input, outputs=[label_output, prob_plot])
        sample_btn.click(random_sample, outputs=[text_input, label_output])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
