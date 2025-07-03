# MoE-KGC-Franka: 基于专家混合模型进行Franka人机交互知识图谱构建

本项目实现了一种新颖的专家混合（MoE）模型，用于在Franka机器人人机交互的背景下，从多模态数据中构建知识图谱。这一技术将为Franka机械臂在具身智能领域的应用提供全新范式。

## 特性
- **多模态数据处理**：处理PDF文档、CSV传感器数据和YAML配置文件
- **专家网络**：针对不同方面的专业专家：
  - 动作专家：机器人动作识别与预测
  - 空间专家：空间推理与关系
  - 时间专家：时间序列建模
  - 语义专家：语义理解
  - 安全专家：安全评估与约束
- **自适应门控**：基于输入动态选择专家
- **图神经网络**：用于知识图谱推理的增强GNN层
- **多任务支持**：支持链接预测、实体分类和关系提取
- **mini-batch子图训练**：采用PyG的NeighborLoader，支持大规模图的高效训练
- **动态专家特征投影**：所有专家网络均采用dynamic_projection层，自动适配输入特征维度

## 安装
### 环境要求
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3（GPU支持）
- torch-geometric >= 2.0
- 其它依赖见 `requirements.txt`

### 安装步骤
```bash
# 克隆项目仓库
git clone https://github.com/alknsk/MoE-KGC-Franka 
cd MoE-franka

# 创建虚拟环境（Linux/Mac）
python -m venv venv
source venv/bin/activate

# Windows环境激活命令
# venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .
```

## 使用流程

### 1. 数据准备与预处理

请确保你的数据目录结构如下：

```
data/
  train/
    *.csv / *.yaml / *.pdf ...
  val/
    ...
  test/
    ...
```

**只需运行如下命令将数据转换为PyG格式：**
```bash
python scripts/prepare_pyg_data.py --data_dir ./data --config config/default_config.yaml
```
该脚本会自动将 `train/val/test` 子目录下的数据转换为 PyG 格式，并保存为 `pyg_data.pt`。

> ⚠️ `scripts/prepare_data.py` 仅用于原始数据的初步整理和分割，常规训练流程无需运行。

### 2. 训练与评估

推荐使用一键脚本：

```bash
bash scripts/run_batch_training.sh
```

该脚本会自动完成数据检查、PyG预处理、mini-batch训练与评估。训练日志与模型保存在 `experiments/` 目录下。

- 训练主入口为 `scripts/train_e2e.py`，支持命令行参数自定义。

### 3. 结果查看

训练输出存储于 `experiments/<exp_name>/`，包括模型检查点、训练日志、训练历史、最终结果、可视化等。

## 主要特性

- **mini-batch子图训练**：采用 PyG 的 NeighborLoader，支持大规模图的高效训练。
- **动态专家特征投影**：所有专家网络均采用 dynamic_projection 层，自动适配输入特征维度。
- **多模态输入**：支持文本、表格、结构化等多种模态。
- **门控机制**：自适应选择最优专家组合。
- **丰富的评估与可视化**：自动生成训练曲线、知识图谱可视化等。

## 常见问题

- **Q: 训练时报错找不到 pyg_data.pt？**
  - 请先运行 `prepare_pyg_data.py` 进行数据预处理。

- **Q: 如何自定义 batch 大小、采样邻居数等？**
  - 修改 `config/default_config.yaml` 或在 `run_batch_training.sh` 中设置参数。

- **Q: 训练入口是哪个？**
  - 推荐用 `bash scripts/run_batch_training.sh`，其核心训练逻辑在 `scripts/train_e2e.py`。

## 其它说明

- 若需自定义数据预处理流程，可参考 `scripts/prepare_data.py`。
- 评估与可视化结果自动保存在 `experiments/<exp_name>/` 目录下。

## 联系方式

如有问题请提交 issue 或联系作者。