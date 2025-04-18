# Enzyme Commission Number Prediction Model

基于序列信息的酶委员会编号（EC Number）预测深度学习模型，结合ProtBERT预训练模型提取蛋白质序列特征，并通过层次化Transformer架构实现多层次EC编号分类。

## 项目亮点
- **特征提取**: 使用ProtBERT对蛋白质序列进行嵌入表示，捕获氨基酸层级语义信息。
- **层次化建模**: 针对EC编号的层次结构（如`a.b.c.d`），设计层级Transformer分类器，逐层细化预测结果。

## 安装

### 环境要求
- Python 3.8+
- PyTorch 1.12+ (需匹配CUDA版本)
- HuggingFace Transformers 4.28+
- 如需直接使用这里提供的权重文件（best_hierarchical_transformer_ec_classifier.pt）和训练日志文件（label_encoders.pkl），其原始环境基于，ubuntu22.04,cuda12.1,pytorch2.5.0,python3.12

### 步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/ooszm/EC-number.git
   cd EC-model
   
2.优先创建虚拟环境，下载对应的模块。

3.可以直接运行transformer文件得到基于本地环境的权重与训练日志，亦或使用这里提供的，然后pret.py是对整个未知序列文件的全体预测，pret2.py则是通过交互方式得到单个序列信息的预测结果。
