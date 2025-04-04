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

### 步骤
1. 克隆仓库：
   ```bash
   git clone https://github.com/ooszm/EC-number.git
   cd EC-model
