# MoE-KGC-Franka: Mixture of Experts for Knowledge Graph Construction in Human-Robot Interaction

This project implements a novel Mixture of Experts (MoE) model for constructing knowledge graphs from multimodal data in the context of Franka robot human-robot interaction.

## Features

- **Multimodal Data Processing**: Handles PDF documents, CSV sensor data, and YAML configuration files
- **Expert Networks**: Specialized experts for different aspects:
  - Action Expert: Robot action recognition and prediction
  - Spatial Expert: Spatial reasoning and relationships
  - Temporal Expert: Temporal sequence modeling
  - Semantic Expert: Semantic understanding
  - Safety Expert: Safety assessment and constraints
- **Adaptive Gating**: Dynamic expert selection based on input
- **Graph Neural Networks**: Enhanced GNN layers for knowledge graph reasoning
- **Multiple Tasks**: Supports link prediction, entity classification, and relation extraction

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3 (for GPU support)
# MoE-KGC-Franka: 人机交互中用于知识图谱构建的专家混合模型

本项目实现了一种新颖的专家混合（MoE）模型，用于在Franka机器人人机交互的背景下，从多模态数据中构建知识图谱。

## 特性
- **多模态数据处理**：处理PDF文档、CSV传感器数据和YAML配置文件。
- **专家网络**：针对不同方面的专业专家：
  - 动作专家：机器人动作识别与预测。
  - 空间专家：空间推理与关系。
  - 时间专家：时间序列建模。
  - 语义专家：语义理解。
  - 安全专家：安全评估与约束。
- **自适应门控**：基于输入动态选择专家。
- **图神经网络**：用于知识图谱推理的增强GNN层。
- **多任务支持**：支持链接预测、实体分类和关系提取。

## 安装
### 要求
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3（用于GPU支持）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/yourusername/moe-kgc-franka.git
cd moe-kgc-franka

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上：venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装包
pip install -e .

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/moe-kgc-franka.git
cd moe-kgc-franka

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

