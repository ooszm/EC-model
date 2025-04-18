#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import torch
import re
import math
import numpy as np
import sys
sys.modules['numpy._core'] = np.core
from transformers import BertTokenizer, BertModel
import torch.nn as nn

# —— 1. 配置默认路径 —— #
MODEL_PATH = r'E:\final\best_hierarchical_transformer_ec_classifier.pt'
LABEL_ENCODERS_PATH = r'E:\final\label_encoders.pkl'

# —— 2. 加载预训练 ProtBERT —— #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> 使用设备: {device}")

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
bert_model.eval()

# —— 3. 特征提取 —— #
def extract_features(sequence: str):
    # 序列预处理
    seq = re.sub(r"[UZOB]", "X", sequence)
    seq = " ".join(seq)
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output, outputs.last_hidden_state

# —— 4. 定义层次化分类器 —— #
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model, self.num_heads = d_model, num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    def split_heads(self, x):
        b, l = x.size(0), x.size(1)
        x = x.view(b, l, self.num_heads, self.d_k).transpose(1,2)
        return x
    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        q, k, v = self.split_heads(self.W_q(q)), self.split_heads(self.W_k(k)), self.split_heads(self.W_v(v))
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v).transpose(1,2).contiguous().view(b, -1, self.d_model)
        return self.W_o(context), weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop1(a))
        f = self.ff(x)
        x = self.norm2(x + self.drop2(f))
        return x

class HierarchicalTransformerClassifier(nn.Module):
    def __init__(self, input_dim=1024, d_model=256, num_heads=4, d_ff=512, num_layers=2, dropout=0.3, num_classes=None):
        super().__init__()
        # 加载 label_encoders 可以得到每层类别数
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
        self.num_classes = [len(enc.classes_) for enc in encoders]
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.global_attn = nn.Sequential(nn.Linear(d_model,1), nn.Softmax(dim=1))
        # 四个层次的分类头
        self.ec1 = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, self.num_classes[0]))
        self.ec2 = nn.Sequential(nn.Linear(d_model+self.num_classes[0], d_model), nn.LayerNorm(d_model),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, self.num_classes[1]))
        self.ec3 = nn.Sequential(nn.Linear(d_model+self.num_classes[1], d_model), nn.LayerNorm(d_model),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, self.num_classes[2]))
        self.ec4 = nn.Sequential(nn.Linear(d_model+self.num_classes[2], d_model), nn.LayerNorm(d_model),
                                 nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, self.num_classes[3]))

    def forward(self, pooled, seq_out):
        x = self.input_proj(seq_out)
        for layer in self.layers:
            x = layer(x)
        attn = self.global_attn(x)               # [B, L, 1]
        global_repr = torch.bmm(attn.transpose(1,2), x).squeeze(1)  # [B, d_model]
        e1 = self.ec1(global_repr); p1 = torch.softmax(e1, dim=-1)
        e2 = self.ec2(torch.cat([global_repr, p1], dim=-1)); p2 = torch.softmax(e2, dim=-1)
        e3 = self.ec3(torch.cat([global_repr, p2], dim=-1)); p3 = torch.softmax(e3, dim=-1)
        e4 = self.ec4(torch.cat([global_repr, p3], dim=-1))
        return [e1, e2, e3, e4]

def load_model():
    model = HierarchicalTransformerClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

def predict_ec(sequence, model, encoders):
    pooled, seq_out = extract_features(sequence)
    logits = model(pooled, seq_out)
    preds = [torch.argmax(l, dim=1).item() for l in logits]
    # 解码成字符串
    ec_parts = []
    for i, idx in enumerate(preds):
        cls = encoders[i].inverse_transform([idx])[0]
        ec_parts.append(cls)
    return ".".join(ec_parts)

def main():
    # 预加载模型和编码器
    model = load_model()
    with open(LABEL_ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)

    # 提示用户输入
    seq = input(">>> 请输入氨基酸序列：\n").strip()
    if not seq:
        print("未输入有效序列，程序退出。")
        return

    ec = predict_ec(seq, model, encoders)
    print(f"\n预测结果：{ec}")

if __name__ == "__main__":
    main()
