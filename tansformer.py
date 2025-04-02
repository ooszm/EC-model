from transformers import BertModel, BertTokenizer
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import math
import numpy as np
    
# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 加载数据
data_path = "E:/down/human/alldata/reviewed_human.tsv"
data = pd.read_csv(data_path, sep="\t")

# 提取 EC number 列中的第一个 EC 编号并分割为四层
data["EC_first"] = data["EC number"].str.split(';').str[0]
labels = data["EC_first"].str.split(".", expand=True).reindex(columns=range(4), fill_value="-")

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
bert_model.to(device)
bert_model.eval()

# 创建自定义数据集类
class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # 预处理序列
        sequence = re.sub(r"[UZOB]", "X", sequence)
        sequence = ' '.join(sequence)
        return sequence, [label[idx] for label in self.labels]
    
    @staticmethod
    def collate_fn(batch):
        sequences, labels = zip(*batch)
        # 对序列进行分词
        inputs = tokenizer(list(sequences), padding=True, truncation=True, return_tensors="pt", max_length=512)
        # 将标签转换为张量
        label_tensors = [torch.tensor(label) for label in zip(*labels)]
        return inputs, label_tensors

# 标签编码
label_encoders = [LabelEncoder() for _ in range(4)]
encoded_labels = []

for i in range(4):
    current_labels = labels[i].values
    encoded_labels.append(torch.tensor(label_encoders[i].fit_transform(current_labels), dtype=torch.long))

# 创建数据集和数据加载器
dataset = ProteinDataset(data["Sequence"].values, encoded_labels)

# 划分训练集和验证集 (80% 训练, 20% 验证)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 16  # 根据GPU内存调整
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ProteinDataset.collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=ProteinDataset.collate_fn)

# 特征提取函数 - 使用pooled输出和序列表示
def extract_features(inputs):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # 返回池化输出 [batch_size, hidden_size] 和最后一层的隐藏状态 [batch_size, sequence_length, hidden_size]
    return outputs.pooler_output, outputs.last_hidden_state

# 多头自注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x):
        # x shape: [batch_size, seq_length, d_model]
        batch_size, seq_length = x.size(0), x.size(1)
        
        # [batch_size, seq_length, num_heads, d_k]
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
        
        # [batch_size, num_heads, seq_length, d_k]
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)
        
        # 分割为多头
        q = self.split_heads(q)  # [batch_size, num_heads, seq_length_q, d_k]
        k = self.split_heads(k)  # [batch_size, num_heads, seq_length_k, d_k]
        v = self.split_heads(v)  # [batch_size, num_heads, seq_length_v, d_k]
        
        # 注意力计算
        # [batch_size, num_heads, seq_length_q, seq_length_k]
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # [batch_size, num_heads, seq_length_q, seq_length_k]
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # [batch_size, num_heads, seq_length_q, d_k]
        context = torch.matmul(attention_weights, v)
        
        # [batch_size, seq_length_q, num_heads, d_k]
        context = context.transpose(1, 2).contiguous()
        
        # [batch_size, seq_length_q, d_model]
        context = context.view(batch_size, -1, self.d_model)
        
        # 最终的线性变换
        output = self.W_o(context)
        
        return output, attention_weights

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # [batch_size, seq_length, d_model] -> [batch_size, seq_length, d_ff]
        x = self.dropout(torch.relu(self.linear1(x)))
        # [batch_size, seq_length, d_ff] -> [batch_size, seq_length, d_model]
        x = self.linear2(x)
        return x

# Transformer编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力机制
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

# 层次化Transformer分类器
class HierarchicalTransformerClassifier(nn.Module):
    def __init__(self, input_dim=1024, d_model=256, num_heads=4, d_ff=512, num_layers=2, dropout=0.3):
        super(HierarchicalTransformerClassifier, self).__init__()
        
        # 获取每个层次的类别数量
        self.num_classes = [len(encoder.classes_) for encoder in label_encoders]
        
        # 降维层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局序列表示
        self.global_attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # 层次化分类器
        # 第一层分类器
        self.ec1_classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.num_classes[0])
        )
        
        # 第二层分类器，将第一层的表示和预测结合
        self.ec2_classifier = nn.Sequential(
            nn.Linear(d_model + self.num_classes[0], d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.num_classes[1])
        )
        
        # 第三层分类器，将第二层的表示和预测结合
        self.ec3_classifier = nn.Sequential(
            nn.Linear(d_model + self.num_classes[1], d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.num_classes[2])
        )
        
        # 第四层分类器，将第三层的表示和预测结合
        self.ec4_classifier = nn.Sequential(
            nn.Linear(d_model + self.num_classes[2], d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.num_classes[3])
        )
    
    def forward(self, pooled_output, sequence_output):
        # 处理序列输出
        # sequence_output: [batch_size, seq_length, input_dim]
        sequence_repr = self.input_proj(sequence_output)
        
        # 通过Transformer编码器层
        for layer in self.encoder_layers:
            sequence_repr = layer(sequence_repr)
        
        # 全局序列表示
        # [batch_size, seq_length, 1]
        attention_weights = self.global_attention(sequence_repr)
        # [batch_size, 1, seq_length] x [batch_size, seq_length, d_model] -> [batch_size, 1, d_model]
        global_repr = torch.bmm(attention_weights.transpose(1, 2), sequence_repr)
        # [batch_size, d_model]
        global_repr = global_repr.squeeze(1)
        
        # 层次化预测
        # 第一层EC预测
        ec1_logits = self.ec1_classifier(global_repr)
        ec1_probs = torch.softmax(ec1_logits, dim=-1)
        
        # 第二层EC预测，考虑第一层的结果
        ec2_input = torch.cat([global_repr, ec1_probs], dim=-1)
        ec2_logits = self.ec2_classifier(ec2_input)
        ec2_probs = torch.softmax(ec2_logits, dim=-1)
        
        # 第三层EC预测，考虑第二层的结果
        ec3_input = torch.cat([global_repr, ec2_probs], dim=-1)
        ec3_logits = self.ec3_classifier(ec3_input)
        ec3_probs = torch.softmax(ec3_logits, dim=-1)
        
        # 第四层EC预测，考虑第三层的结果
        ec4_input = torch.cat([global_repr, ec3_probs], dim=-1)
        ec4_logits = self.ec4_classifier(ec4_input)
        
        return [ec1_logits, ec2_logits, ec3_logits, ec4_logits]

# 初始化层次化Transformer分类器
classifier = HierarchicalTransformerClassifier()
classifier.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 训练函数
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = [0] * 4
    total = 0
    
    for inputs, labels in tqdm(data_loader, desc="Training"):
        # 提取特征
        pooled_output, sequence_output = extract_features(inputs)
        
        # 前向传播
        outputs = model(pooled_output, sequence_output)
        
        # 计算损失
        loss = sum(criterion(output, label.to(device)) for output, label in zip(outputs, labels))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total += labels[0].size(0)
        
        # 计算每层的准确率
        for i, (output, label) in enumerate(zip(outputs, labels)):
            pred = output.argmax(dim=1).cpu()
            correct[i] += (pred == label).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(data_loader)
    accuracies = [c / total for c in correct]
    
    return avg_loss, accuracies

# 验证函数
def validate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = [0] * 4
    total = 0
    all_preds = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Validation"):
            # 提取特征
            pooled_output, sequence_output = extract_features(inputs)
            
            # 前向传播
            outputs = model(pooled_output, sequence_output)
            
            # 计算损失
            loss = sum(criterion(output, label.to(device)) for output, label in zip(outputs, labels))
            
            # 统计
            total_loss += loss.item()
            total += labels[0].size(0)
            
            # 计算每层的准确率
            for i, (output, label) in enumerate(zip(outputs, labels)):
                pred = output.argmax(dim=1).cpu()
                correct[i] += (pred == label).sum().item()
                all_preds[i].extend(pred.tolist())
                all_labels[i].extend(label.tolist())
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(data_loader)
    accuracies = [c / total for c in correct]
    
    return avg_loss, accuracies, all_preds, all_labels

# 训练模型
epochs = 30
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(epochs):
    # 训练
    train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, criterion)
    
    # 验证
    val_loss, val_acc, _, _ = validate(classifier, val_loader, criterion)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 输出训练信息
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}, Accuracies: EC1={train_acc[0]:.4f}, EC2={train_acc[1]:.4f}, EC3={train_acc[2]:.4f}, EC4={train_acc[3]:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Accuracies: EC1={val_acc[0]:.4f}, EC2={val_acc[1]:.4f}, EC3={val_acc[2]:.4f}, EC4={val_acc[3]:.4f}")
    
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # 保存最佳模型
        torch.save(classifier.state_dict(), "best_hierarchical_transformer_ec_classifier.pt")
        print("Model saved!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

# 加载最佳模型进行预测
classifier.load_state_dict(torch.load("best_hierarchical_transformer_ec_classifier.pt"))
classifier.eval()

# 最终测试
test_loss, test_acc, test_preds, test_labels = validate(classifier, val_loader, criterion)
print(f"Final Test: Loss={test_loss:.4f}, Accuracies: EC1={test_acc[0]:.4f}, EC2={test_acc[1]:.4f}, EC3={test_acc[2]:.4f}, EC4={test_acc[3]:.4f}")

# 计算混淆矩阵和其他指标
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 为每个EC层级生成详细的评估报告
for i in range(4):
    level_name = f"EC{i+1}"
    
    # 获取测试集中实际出现的唯一类别
    unique_classes = sorted(list(set(test_labels[i])))
    class_names = label_encoders[i].inverse_transform(unique_classes)
    
    print(f"\n=== {level_name} 分类报告 ===")
    
    # 使用适当的标签生成分类报告
    report = classification_report(
        test_labels[i], 
        test_preds[i], 
        labels=unique_classes,  # 指定要包含在报告中的标签
        target_names=class_names if len(class_names) < 10 else None
    )
    print(report)
    
    # 如果类别数量较少，绘制混淆矩阵
    if len(unique_classes) < 10:
        cm = confusion_matrix(test_labels[i], test_preds[i], labels=unique_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{level_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{level_name}_confusion_matrix.png')
        plt.close()

# 反向映射函数：将预测的数字索引转换回EC号
def map_predictions_to_ec(predictions):
    ec_predictions = []
    for i, pred in enumerate(predictions):
        ec_level = label_encoders[i].inverse_transform(pred)
        ec_predictions.append(ec_level)
    
    # 组合四个层次的预测为完整EC号
    full_ec = []
    for i in range(len(ec_predictions[0])):
        ec = ".".join(str(ec_predictions[j][i]) for j in range(4))
        ec = ec.replace(".-", ".-")  # 处理缺失的层次
        full_ec.append(ec)
    
    return full_ec

# 保存模型和标签编码器
import pickle
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("训练完成！")
