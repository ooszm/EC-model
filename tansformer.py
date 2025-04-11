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
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 创建记录指标的目录和文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"training_logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# 创建指标记录字典
training_history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': {f'ec{i+1}': [] for i in range(4)},
    'val_acc': {f'ec{i+1}': [] for i in range(4)},
    'learning_rate': []
}

# 创建批次级别的指标追踪
batch_metrics = {
    'train_loss': [],
    'batch_idx': []
}

# 加载数据
data_path = "./train_data.tsv"
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

# 明确冻结ProtBERT模型的所有参数
for param in bert_model.parameters():
    param.requires_grad = False


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

batch_size = 64  # 根据GPU内存调整
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

# 保存当前的学习率
def get_lr():
    for param_group in optimizer.param_groups:
        return param_group['lr']

# 训练函数
def train_epoch(model, data_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = [0] * 4
    total = 0
    batch_count = 0
    
    for i, (inputs, labels) in enumerate(tqdm(data_loader, desc=f"Training Epoch {epoch+1}")):
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
        
        # 记录每个批次的损失
        batch_count += 1
        if batch_count % 5 == 0:  # 每5个批次记录一次
            batch_metrics['train_loss'].append(loss.item())
            batch_metrics['batch_idx'].append(batch_count + epoch * len(data_loader))
    
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

# 绘制训练历史记录
def plot_training_history(history, filename):
    # 创建图表：损失、训练准确率、验证准确率
    plt.figure(figsize=(15, 15))
    
    # 绘制损失
    plt.subplot(3, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制训练集准确率
    plt.subplot(3, 1, 2)
    for i, level in enumerate(['ec1', 'ec2', 'ec3', 'ec4']):
        plt.plot(history['train_acc'][level], label=f'EC{i+1}')
    plt.title('Training Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制验证集准确率
    plt.subplot(3, 1, 3)
    for i, level in enumerate(['ec1', 'ec2', 'ec3', 'ec4']):
        plt.plot(history['val_acc'][level], label=f'EC{i+1}')
    plt.title('Validation Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
    # 单独为每个EC级别绘制单独的准确率趋势图
    plt.figure(figsize=(15, 10))
    for i, level in enumerate(['ec1', 'ec2', 'ec3', 'ec4']):
        plt.subplot(2, 2, i+1)  # 2x2网格，每个EC一个子图
        plt.plot(history['train_acc'][level], label=f'Training Accuracy')
        plt.plot(history['val_acc'][level], label=f'Validation Accuracy')
        plt.title(f'EC{i+1} Accuracy over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '_per_level.png'))
    plt.close()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# 绘制批次级别的损失
def plot_batch_loss(batch_metrics, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(batch_metrics['batch_idx'], batch_metrics['train_loss'])
    plt.title('Training Loss per Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

# 训练模型
epochs = 30
best_val_loss = float('inf')
patience = 10
counter = 0

# 记录开始时间
start_time = datetime.now()
print(f"开始训练时间: {start_time}")

for epoch in range(epochs):
    epoch_start_time = datetime.now()
    
    # 训练
    train_loss, train_acc = train_epoch(classifier, train_loader, optimizer, criterion, epoch)
    
    # 验证
    val_loss, val_acc, _, _ = validate(classifier, val_loader, criterion)
    
    # 学习率调整
    current_lr = get_lr()
    scheduler.step(val_loss)
    
    # 记录指标
    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    for i in range(4):
        training_history['train_acc'][f'ec{i+1}'].append(train_acc[i])
        training_history['val_acc'][f'ec{i+1}'].append(val_acc[i])
    training_history['learning_rate'].append(current_lr)
    
    # 只在最后保存训练历史记录文件
    if epoch == epochs - 1 or counter >= patience:
        with open(f"{log_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=4)
        
        # 绘制最终训练历史图表
        plot_training_history(training_history, f"{log_dir}/training_history.png")
        
        # 绘制批次级别的损失
        plot_batch_loss(batch_metrics, f"{log_dir}/batch_loss.png")
    
    # 计算epoch花费的时间
    epoch_time = datetime.now() - epoch_start_time
    
    # 输出训练信息
    print(f"Epoch {epoch+1}/{epochs} (用时: {epoch_time})")
    print(f"Train Loss: {train_loss:.4f}, Accuracies: EC1={train_acc[0]:.4f}, EC2={train_acc[1]:.4f}, EC3={train_acc[2]:.4f}, EC4={train_acc[3]:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Accuracies: EC1={val_acc[0]:.4f}, EC2={val_acc[1]:.4f}, EC3={val_acc[2]:.4f}, EC4={val_acc[3]:.4f}")
    print(f"Learning Rate: {current_lr}")
    
    # 保存详细的epoch日志
    epoch_log = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': {f'ec{i+1}': train_acc[i] for i in range(4)},
        'val_acc': {f'ec{i+1}': val_acc[i] for i in range(4)},
        'learning_rate': current_lr,
        'epoch_time': str(epoch_time)
    }
    
    # 只保存最后一轮的详细日志
    if epoch == epochs - 1 or counter >= patience:
        with open(f"{log_dir}/final_epoch_log.json", 'w') as f:
            json.dump(epoch_log, f, indent=4)
    
    # 早停
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # 保存模型
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, f"{log_dir}/best_model_epoch_{epoch+1}.pt")
        
        # 额外保存一个固定名称的最佳模型
        torch.save(classifier.state_dict(), f"{log_dir}/best_hierarchical_transformer_ec_classifier.pt")
        print("Model saved!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

# 记录总训练时间
total_training_time = datetime.now() - start_time
print(f"总训练时间: {total_training_time}")

# 记录最终的配置和性能摘要
summary = {
    'total_epochs': epoch + 1,
    'best_val_loss': best_val_loss,
    'final_train_loss': train_loss,
    'final_train_acc': {f'ec{i+1}': train_acc[i] for i in range(4)},
    'final_val_loss': val_loss,
    'final_val_acc': {f'ec{i+1}': val_acc[i] for i in range(4)},
    'total_training_time': str(total_training_time),
    'batch_size': batch_size,
    'model_config': {
        'input_dim': 1024,
        'd_model': 256,
        'num_heads': 4,
        'd_ff': 512,
        'num_layers': 2,
        'dropout': 0.3
    }
}

with open(f"{log_dir}/training_summary.json", 'w') as f:
    json.dump(summary, f, indent=4)

# 加载最佳模型进行预测
best_model_path = f"{log_dir}/best_hierarchical_transformer_ec_classifier.pt"
classifier.load_state_dict(torch.load(best_model_path))
classifier.eval()

# 最终测试
test_loss, test_acc, test_preds, test_labels = validate(classifier, val_loader, criterion)
print(f"Final Test: Loss={test_loss:.4f}, Accuracies: EC1={test_acc[0]:.4f}, EC2={test_acc[1]:.4f}, EC3={test_acc[2]:.4f}, EC4={test_acc[3]:.4f}")

# 计算混淆矩阵和其他指标
from sklearn.metrics import classification_report, confusion_matrix

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
    
    # 保存分类报告到文件
    with open(f"{log_dir}/{level_name}_classification_report.txt", 'w') as f:
        f.write(report)
    
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
        plt.savefig(f"{log_dir}/{level_name}_confusion_matrix.png")
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

# 生成预测与实际EC号的对比表格
val_sequences = [dataset[val_dataset.indices[i]][0] for i in range(len(val_dataset))]
predicted_ec = map_predictions_to_ec(test_preds)

# 获取真实的EC号
true_ec = []
for i in range(len(val_dataset)):
    idx = val_dataset.indices[i]
    true_ec_parts = [label_encoders[j].inverse_transform([encoded_labels[j][idx].item()])[0] for j in range(4)]
    true_ec.append(".".join(str(part) for part in true_ec_parts))

# 创建结果DataFrame
results_df = pd.DataFrame({
    'Sequence': val_sequences,
    'True_EC': true_ec,
    'Predicted_EC': predicted_ec,
    'Correct': [t == p for t, p in zip(true_ec, predicted_ec)]
})

# 保存结果到CSV
results_df.to_csv(f"{log_dir}/prediction_results.csv", index=False)

# 保存模型和标签编码器
import pickle
with open(f"{log_dir}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# 创建一个交互式的预测结果分析函数
def analyze_prediction_accuracy():
    # 按照EC层级计算准确率
    ec_level_accuracy = []
    for i in range(4):
        correct = 0
        total = len(test_labels[i])
        for pred, true in zip(test_preds[i], test_labels[i]):
            if pred == true:
                correct += 1
        ec_level_accuracy.append(correct / total)
    
    # 计算完全匹配的准确率 (所有四个层级都正确)
    complete_match = sum([t == p for t, p in zip(true_ec, predicted_ec)])
    complete_accuracy = complete_match / len(true_ec)
    
    # 创建分析摘要
    analysis = {
        'ec_level_accuracy': {f'EC{i+1}': ec_level_accuracy[i] for i in range(4)},
        'complete_match_accuracy': complete_accuracy,
        'total_test_samples': len(true_ec),
        'correct_predictions': complete_match
    }
    
    # 保存分析结果
    with open(f"{log_dir}/prediction_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=4)
    
    # 绘制准确率图表
    plt.figure(figsize=(10, 6))
    levels = [f'EC{i+1}' for i in range(4)]
    levels.append('Complete')
    accuracies = ec_level_accuracy + [complete_accuracy]
    
    bars = plt.bar(levels, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    
    # 在柱状图上添加具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Prediction Accuracy by EC Level')
    plt.xlabel('EC Level')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)  # 设置y轴范围
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{log_dir}/ec_level_accuracy.png")
    plt.close()
    
    # 按照第一层EC号分类的准确率分析
    ec1_groups = {}
    for t_ec, p_ec in zip(true_ec, predicted_ec):
        ec1 = t_ec.split('.')[0]
        if ec1 not in ec1_groups:
            ec1_groups[ec1] = {'total': 0, 'correct': 0}
        
        ec1_groups[ec1]['total'] += 1
        if t_ec == p_ec:
            ec1_groups[ec1]['correct'] += 1
    
    # 计算每个EC1组的准确率
    ec1_accuracy = {ec1: group['correct']/group['total'] for ec1, group in ec1_groups.items() if group['total'] >= 5}
    
    # 绘制EC1组的准确率
    if ec1_accuracy:
        plt.figure(figsize=(12, 6))
        ec1_keys = sorted(ec1_accuracy.keys())
        ec1_values = [ec1_accuracy[k] for k in ec1_keys]
        
        bars = plt.bar(ec1_keys, ec1_values)
        
        # 在柱状图上添加具体数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', rotation=90)
        
        plt.title('Prediction Accuracy by EC1 Group (min 5 samples)')
        plt.xlabel('EC1')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{log_dir}/ec1_group_accuracy.png")
        plt.close()
    
    return analysis

# 运行分析
analysis_results = analyze_prediction_accuracy()
print("预测分析完成！分析结果已保存到:", f"{log_dir}/prediction_analysis.json")

# 创建训练过程动态可视化
def create_dynamic_visualization():
    # 提取训练历史数据
    epochs = list(range(1, len(training_history['train_loss']) + 1))
    train_loss = training_history['train_loss']
    val_loss = training_history['val_loss']
    
    # 创建一个HTML动态可视化报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>模型训练过程分析</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .chart {{
                margin-bottom: 30px;
            }}
            h1, h2 {{
                color: #333;
            }}
            .summary {{
                background-color: #f9f9f9;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>层次化Transformer蛋白质EC号分类模型训练分析</h1>
            
            <div class="summary">
                <h2>训练摘要</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>值</th>
                    </tr>
                    <tr>
                        <td>总训练轮次</td>
                        <td>{len(epochs)}</td>
                    </tr>
                    <tr>
                        <td>最佳验证损失</td>
                        <td>{best_val_loss:.4f}</td>
                    </tr>
                    <tr>
                        <td>最终训练损失</td>
                        <td>{train_loss[-1]:.4f}</td>
                    </tr>
                    <tr>
                        <td>最终验证损失</td>
                        <td>{val_loss[-1]:.4f}</td>
                    </tr>
                    <tr>
                        <td>EC1 准确率</td>
                        <td>{test_acc[0]:.4f}</td>
                    </tr>
                    <tr>
                        <td>EC2 准确率</td>
                        <td>{test_acc[1]:.4f}</td>
                    </tr>
                    <tr>
                        <td>EC3 准确率</td>
                        <td>{test_acc[2]:.4f}</td>
                    </tr>
                    <tr>
                        <td>EC4 准确率</td>
                        <td>{test_acc[3]:.4f}</td>
                    </tr>
                    <tr>
                        <td>完全匹配准确率</td>
                        <td>{analysis_results['complete_match_accuracy']:.4f}</td>
                    </tr>
                    <tr>
                        <td>总训练时间</td>
                        <td>{total_training_time}</td>
                    </tr>
                </table>
            </div>
            
            <div class="chart">
                <h2>损失曲线</h2>
                <div id="lossChart"></div>
            </div>
            
            <div class="chart">
                <h2>学习率变化</h2>
                <div id="lrChart"></div>
            </div>
            
            <div class="chart">
                <h2>准确率曲线</h2>
                <div id="accChart"></div>
            </div>
            
            <div class="chart">
                <h2>批次级别损失</h2>
                <div id="batchLossChart"></div>
            </div>
        </div>
        
        <script>
            // 损失曲线
            var lossData = [
                {{
                    x: {epochs},
                    y: {train_loss},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '训练损失'
                }},
                {{
                    x: {epochs},
                    y: {val_loss},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '验证损失'
                }}
            ];
            
            var lossLayout = {{
                title: '训练和验证损失',
                xaxis: {{
                    title: '轮次'
                }},
                yaxis: {{
                    title: '损失'
                }},
                hovermode: 'closest'
            }};
            
            Plotly.newPlot('lossChart', lossData, lossLayout);
            
            // 学习率曲线
            var lrData = [
                {{
                    x: {epochs},
                    y: {training_history['learning_rate']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '学习率'
                }}
            ];
            
            var lrLayout = {{
                title: '学习率变化',
                xaxis: {{
                    title: '轮次'
                }},
                yaxis: {{
                    title: '学习率'
                }},
                hovermode: 'closest'
            }};
            
            Plotly.newPlot('lrChart', lrData, lrLayout);
            
            // 准确率曲线
            var accData = [
                {{
                    x: {epochs},
                    y: {training_history['train_acc']['ec1']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC1 训练'
                }},
                {{
                    x: {epochs},
                    y: {training_history['val_acc']['ec1']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC1 验证'
                }},
                {{
                    x: {epochs},
                    y: {training_history['train_acc']['ec2']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC2 训练'
                }},
                {{
                    x: {epochs},
                    y: {training_history['val_acc']['ec2']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC2 验证'
                }},
                {{
                    x: {epochs},
                    y: {training_history['train_acc']['ec3']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC3 训练'
                }},
                {{
                    x: {epochs},
                    y: {training_history['val_acc']['ec3']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC3 验证'
                }},
                {{
                    x: {epochs},
                    y: {training_history['train_acc']['ec4']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC4 训练'
                }},
                {{
                    x: {epochs},
                    y: {training_history['val_acc']['ec4']},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'EC4 验证'
                }}
            ];
            
            var accLayout = {{
                title: '各层级准确率',
                xaxis: {{
                    title: '轮次'
                }},
                yaxis: {{
                    title: '准确率'
                }},
                hovermode: 'closest'
            }};
            
            Plotly.newPlot('accChart', accData, accLayout);
            
            // 批次损失曲线
            var batchLossData = [
                {{
                    x: {batch_metrics['batch_idx']},
                    y: {batch_metrics['train_loss']},
                    type: 'scatter',
                    mode: 'lines',
                    name: '批次损失'
                }}
            ];
            
            var batchLossLayout = {{
                title: '批次损失变化',
                xaxis: {{
                    title: '批次'
                }},
                yaxis: {{
                    title: '损失'
                }},
                hovermode: 'closest'
            }};
            
            Plotly.newPlot('batchLossChart', batchLossData, batchLossLayout);
        </script>
    </body>
    </html>
    """
    
    # 保存HTML文件
    with open(f"{log_dir}/training_visualization.html", "w") as f:
        f.write(html_content)
    
    print("动态可视化报告已创建！请打开:", f"{log_dir}/training_visualization.html")

# 创建动态可视化
create_dynamic_visualization()

print(f"训练完成！所有记录和分析已保存到: {log_dir}")
