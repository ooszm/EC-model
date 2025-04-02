import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json
import os
from datetime import datetime
import pickle
import torch.nn as nn
import math
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# 创建评估结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
eval_dir = f"evaluation_results_{timestamp}"
os.makedirs(eval_dir, exist_ok=True)

# 加载标签编码器和模型
with open("./label_encoders.pkl", "rb") as f:  # 替换为你的训练日志目录
    label_encoders = pickle.load(f)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
bert_model.to(device)
bert_model.eval()

# 特征提取函数 - 使用pooled输出和序列表示
def extract_features(inputs):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    # 返回池化输出 [batch_size, hidden_size] 和最后一层的隐藏状态 [batch_size, sequence_length, hidden_size]
    return outputs.pooler_output, outputs.last_hidden_state

# 加载测试数据
test_data_path = "E:/down/human/alldata/reviewed_human.tsv"  # 替换为你的测试数据路径
test_data = pd.read_csv(test_data_path, sep="\t")

# 提取 EC number 列中的第一个 EC 编号并分割为四层
test_data["EC_first"] = test_data["EC number"].str.split(';').str[0]
test_labels = test_data["EC_first"].str.split(".", expand=True).reindex(columns=range(4), fill_value="-")

# 对测试标签进行编码
encoded_test_labels = []
for i in range(4):
    # 注意：这里使用transform而不是fit_transform，因为我们需要使用与训练集相同的编码
    current_labels = test_labels[i].values
    try:
        encoded_test_labels.append(torch.tensor(label_encoders[i].transform(current_labels), dtype=torch.long))
    except ValueError as e:
        # 处理测试集中出现的未知标签
        print(f"警告：EC{i+1}层级存在训练集中未见过的标签。将这些样本标记为特殊值。")
        # 创建标记数组
        encoded = []
        for label in current_labels:
            try:
                encoded.append(label_encoders[i].transform([label])[0])
            except:
                encoded.append(-1)  # -1表示未知类别
        encoded_test_labels.append(torch.tensor(encoded, dtype=torch.long))

# 重用你的ProteinDataset类
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

# 创建测试数据集和数据加载器
test_dataset = ProteinDataset(test_data["Sequence"].values, encoded_test_labels)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=ProteinDataset.collate_fn)

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

# 加载预训练的模型
def load_model(model_path):
    # 创建模型实例
    model = HierarchicalTransformerClassifier()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# 加载你的最佳模型
model_path = "./training_logs_20250328_200554/best_hierarchical_transformer_ec_classifier.pt"  # 替换为你的模型路径
model = load_model(model_path)
 
# 测试集评估函数
def evaluate_test_set(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = [0] * 4
    total = 0
    all_preds = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]
    all_probs = [[] for _ in range(4)]  # 存储每个类别的概率分布
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Test Evaluation"):
            # 提取特征
            pooled_output, sequence_output = extract_features(inputs)
            
            # 前向传播
            outputs = model(pooled_output, sequence_output)
            
            # 计算损失 (忽略未知类别)
            loss = 0
            for i, (output, label) in enumerate(zip(outputs, labels)):
                # 过滤掉未知类别的样本
                valid_idx = label >= 0
                if valid_idx.sum() > 0:
                    loss += criterion(output[valid_idx], label[valid_idx].to(device))
            
            # 统计
            total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            total += labels[0].size(0)
            
            # 计算每层的准确率和收集预测结果
            for i, (output, label) in enumerate(zip(outputs, labels)):
                pred = output.argmax(dim=1).cpu()
                probs = torch.softmax(output, dim=1).cpu().numpy()  # 获取概率分布
                
                # 只考虑有效标签
                valid_idx = label >= 0
                correct[i] += (pred[valid_idx] == label[valid_idx]).sum().item()
                
                all_preds[i].extend(pred.tolist())
                all_labels[i].extend(label.tolist())
                all_probs[i].extend(probs)
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracies = [c / max(valid_count, 1) for c, valid_count in 
                  zip([c for c in correct], 
                      [sum(1 for l in labels if l >= 0) for labels in all_labels])]
    
    return avg_loss, accuracies, all_preds, all_labels, all_probs

# 运行测试集评估
criterion = nn.CrossEntropyLoss()
test_loss, test_acc, test_preds, test_labels, test_probs = evaluate_test_set(model, test_loader, criterion)

# 输出测试结果摘要
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracies: EC1={test_acc[0]:.4f}, EC2={test_acc[1]:.4f}, EC3={test_acc[2]:.4f}, EC4={test_acc[3]:.4f}")

# 保存测试结果
test_results = {
    'test_loss': test_loss,
    'test_acc': {f'ec{i+1}': test_acc[i] for i in range(4)},
    'timestamp': timestamp,
    'model_path': model_path
}

with open(f"{eval_dir}/test_results.json", 'w') as f:
    json.dump(test_results, f, indent=4)

# 为每个EC层级生成详细的评估报告
for i in range(4):
    level_name = f"EC{i+1}"
    
    # 过滤掉未知标签
    valid_indices = [j for j, label in enumerate(test_labels[i]) if label >= 0]
    valid_preds = [test_preds[i][j] for j in valid_indices]
    valid_labels = [test_labels[i][j] for j in valid_indices]
    
    if not valid_indices:
        print(f"\n没有{level_name}层级的有效测试样本")
        continue
    
    # 获取测试集中实际出现的唯一类别
    unique_classes = sorted(list(set(valid_labels)))
    class_names = label_encoders[i].inverse_transform(unique_classes)
    
    print(f"\n=== {level_name} 分类报告 ===")
    
    # 使用适当的标签生成分类报告
    report = classification_report(
        valid_labels, 
        valid_preds, 
        labels=unique_classes,
        target_names=class_names if len(class_names) < 10 else None,
        digits=4
    )
    print(report)
    
    # 保存分类报告到文件
    with open(f"{eval_dir}/{level_name}_classification_report.txt", 'w') as f:
        f.write(report)
    
    # 如果类别数量合适，绘制混淆矩阵
    if 2 <= len(unique_classes) < 20:
        cm = confusion_matrix(valid_labels, valid_preds, labels=unique_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{level_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f"{eval_dir}/{level_name}_confusion_matrix.png")
        plt.close()
    
    # 计算和可视化每个类别的准确率
    class_accuracies = {}
    for cls in unique_classes:
        cls_indices = [i for i, l in enumerate(valid_labels) if l == cls]
        if cls_indices:
            correct = sum(1 for i in cls_indices if valid_preds[i] == valid_labels[i])
            class_accuracies[label_encoders[i].inverse_transform([cls])[0]] = correct / len(cls_indices)
    
    # 按准确率排序
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    classes, accs = zip(*sorted_classes)
    
    # 绘制类别准确率条形图
    plt.figure(figsize=(14, 8))
    bars = plt.bar(classes, accs)
    
    # 在条形上添加准确率值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', rotation=90)
    
    plt.title(f'{level_name} Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{eval_dir}/{level_name}_class_accuracy.png")
    plt.close()
    
    # 分析错误预测
    error_analysis = []
    for idx, (true, pred) in enumerate(zip(valid_labels, valid_preds)):
        if true != pred:
            true_class = label_encoders[i].inverse_transform([true])[0]
            pred_class = label_encoders[i].inverse_transform([pred])[0]
            sample_idx = valid_indices[idx]
            sequence = test_data.iloc[sample_idx]["Sequence"]
            error_analysis.append({
                'sample_idx': sample_idx,
                'sequence': sequence[:20] + "..." if len(sequence) > 20 else sequence,  # 截断序列
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': float(test_probs[i][sample_idx][pred])
            })
    
    # 保存错误分析
    if error_analysis:
        error_df = pd.DataFrame(error_analysis)
        error_df.to_csv(f"{eval_dir}/{level_name}_error_analysis.csv", index=False)

# 生成层次化预测正确率分析
hierarchical_accuracy = []
for i in range(len(test_labels[0])):
    # 获取每层预测是否正确
    correct_by_level = []
    for level in range(4):
        if test_labels[level][i] >= 0:  # 确保是有效标签
            correct_by_level.append(test_preds[level][i] == test_labels[level][i])
        else:
            correct_by_level.append(None)  # 未知标签标记为None
    
    hierarchical_accuracy.append(correct_by_level)

# 计算多种层次化准确率指标
hier_metrics = {
    'exact_match': 0,  # 所有层级都正确
    'partial_match': {  # 部分层级正确
        '0_level': 0,   
        '1_level': 0,
        '2_levels': 0,
        '3_levels': 0
    },
    'level_progression': {  # 层级预测的连续性
        'correct_until_ec1': 0,
        'correct_until_ec2': 0,
        'correct_until_ec3': 0,
        'correct_until_ec4': 0
    }
}

for correct_levels in hierarchical_accuracy:
    # 跳过包含未知标签的样本
    if None in correct_levels:
        continue
    
    correct_count = sum(correct_levels)
    
    # 计算完全匹配
    if correct_count == 4:
        hier_metrics['exact_match'] += 1
    else:
        # 计算部分匹配
        hier_metrics['partial_match'][f'{correct_count}_level' + ('s' if correct_count > 1 else '')] += 1
    
    # 计算连续正确预测
    for level in range(4):
        if all(correct_levels[:level+1]):
            hier_metrics['level_progression'][f'correct_until_ec{level+1}'] += 1

# 计算比例
total_valid_samples = sum(1 for corr in hierarchical_accuracy if None not in corr)
for key in ['exact_match']:
    hier_metrics[key] = hier_metrics[key] / total_valid_samples if total_valid_samples > 0 else 0

for key in ['partial_match', 'level_progression']:
    for subkey in hier_metrics[key]:
        hier_metrics[key][subkey] = hier_metrics[key][subkey] / total_valid_samples if total_valid_samples > 0 else 0

# 保存层次化指标
with open(f"{eval_dir}/hierarchical_metrics.json", 'w') as f:
    json.dump(hier_metrics, f, indent=4)

# 可视化层次化准确率
plt.figure(figsize=(12, 6))
levels = ['EC1', 'EC2', 'EC3', 'EC4', 'Exact Match']
values = [test_acc[0], test_acc[1], test_acc[2], test_acc[3], hier_metrics['exact_match']]

bars = plt.bar(levels, values, color=['blue', 'green', 'orange', 'red', 'purple'])

# 在柱状图上添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

plt.title('Hierarchical Prediction Accuracy')
plt.xlabel('Level')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{eval_dir}/hierarchical_accuracy.png")
plt.close()

# 可视化层级进展准确率
plt.figure(figsize=(10, 6))
progression_levels = list(hier_metrics['level_progression'].keys())
progression_values = [hier_metrics['level_progression'][level] for level in progression_levels]
pretty_labels = ['Up to EC1', 'Up to EC2', 'Up to EC3', 'Up to EC4']

bars = plt.bar(pretty_labels, progression_values, color=['lightblue', 'lightgreen', 'lightsalmon', 'lightcoral'])

# 在柱状图上添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

plt.title('Progressive Accuracy (correct predictions at each level)')
plt.xlabel('Prediction Depth')
plt.ylabel('Proportion of Samples')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(f"{eval_dir}/progressive_accuracy.png")
plt.close()

# 创建综合报告
report_content = f"""
# 测试集评估报告

## 模型信息
- 评估时间: {timestamp}
- 模型路径: {model_path}

## 总体性能
- 测试损失: {test_loss:.4f}
- EC1 准确率: {test_acc[0]:.4f}
- EC2 准确率: {test_acc[1]:.4f}
- EC3 准确率: {test_acc[2]:.4f}
- EC4 准确率: {test_acc[3]:.4f}
- 完全匹配准确率: {hier_metrics['exact_match']:.4f}

## 层次化准确率分析
- 完全匹配 (所有EC层级正确): {hier_metrics['exact_match']:.4f}
- 仅1个层级正确: {hier_metrics['partial_match']['1_level']:.4f}
- 2个层级正确: {hier_metrics['partial_match']['2_levels']:.4f}
- 3个层级正确: {hier_metrics['partial_match']['3_levels']:.4f}

## 连续预测准确率
- 正确预测到EC1: {hier_metrics['level_progression']['correct_until_ec1']:.4f}
- 正确预测到EC2: {hier_metrics['level_progression']['correct_until_ec2']:.4f}
- 正确预测到EC3: {hier_metrics['level_progression']['correct_until_ec3']:.4f}
- 正确预测到EC4 (所有层级): {hier_metrics['level_progression']['correct_until_ec4']:.4f}

## 测试集数据摘要
- 测试样本总数: {len(test_data)}
- 有效样本数 (无未知标签): {total_valid_samples}

## 各层级详细报告
详细的分类报告、混淆矩阵和每个类别的准确率可以在相应的文件中找到:
- EC1: {eval_dir}/EC1_classification_report.txt
- EC2: {eval_dir}/EC2_classification_report.txt
- EC3: {eval_dir}/EC3_classification_report.txt
- EC4: {eval_dir}/EC4_classification_report.txt
"""

with open(f"{eval_dir}/test_evaluation_report.md", 'w') as f:
    f.write(report_content)

print(f"测试评估完成！所有结果已保存到: {eval_dir}")
