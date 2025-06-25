# MoE-KGC-Franka: 人机交互中用于知识图谱构建的专家混合模型

本项目实现了一种新颖的专家混合（MoE）模型，用于在Franka机器人人机交互的背景下，从多模态数据中构建知识图谱。


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


## 安装
### 环境要求
- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3（GPU支持）

### 安装步骤
```bash
# 克隆项目仓库
git clone https://github.com/yourusername/moe-kgc-franka.git
cd moe-kgc-franka

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


# MoE-KGC 模型完整使用手册
## 1. 模型使用流程
### 1.1 数据准备与预处理
```bash
# 执行数据处理脚本
python scripts/prepare_data.py \
    --raw_data ./raw_data \
    --output_dir ./data \
    --split_ratio 0.7 0.15 0.15 \
    --validate --preprocess
```
### 处理内容：
自动扫描raw_data目录下的 PDF/CSV/YAML 文件
按 7:1.5:1.5 比例分割数据集
验证数据格式并构建知识图谱
在data/train/生成可视化文件kg_sample.html
### 1.2 模型训练
```bash
# 基础训练命令
python scripts/train_e2e.py \
    --data_dir ./data \
    --task link_prediction \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --exp_name my_first_experiment \
    --use_wandb
```
### 结果查看
```bash
训练输出存储于experiments/my_first_experiment/：
plaintext
experiments/my_first_experiment/
├── checkpoints/              # 模型检查点
│   ├── best_model.pt        # 最佳模型
│   ├── final_model.pt       # 最终模型
│   └── checkpoint_epoch_10.pt
├── logs/                     # 训练日志
│   └── moe_kgc_20240115_143022.log
├── training_history.json     # 训练历史数据
├── training_history.png      # Loss曲线图
├── final_results.json        # 最终测试结果
├── test_visualization.png    # 测试可视化
├── baseline_comparison.csv   # 基线对比表格
├── comparison_visualization.png  # 对比可视化图
└── knowledge_graph_prediction.html  # 知识图谱可视化
```

### 可视化内容：
Loss 曲线（训练 / 验证）
准确率曲线
F1 分数曲线
混淆矩阵
与 GAT/GIN/SAGE/RGCN 的对比图表
## 2. 训练优化指南
### 2.1 初始结果查看

python
运行
```bash
# 查看最后50行日志
tail -n 50 experiments/my_first_experiment/logs/*.log

# 查看最终结果
cat experiments/my_first_experiment/final_results.json
```

### 2.2 常见问题与调参
#### 问题 1：过拟合
```bash
# 调整参数命令
python scripts/train_e2e.py \
    --data_dir ./data \
    --config config/default_config.yaml \
    --exp_name fix_overfitting

```
配置修改（config/default_config.yaml）：
```bash
model:
  dropout_rate: 0.3  # 从0.1增加
training:
  weight_decay: 1e-4  # 从1e-5增加
data:
  augmentation: true  # 启用数据增强
```
#### 问题 2：收敛缓慢
```bash
# 调整学习率与批次
python scripts/train_e2e.py \
    --data_dir ./data \    
    --lr 0.001 \          # 增大学习率
    --batch_size 64 \     # 增大批次
    --exp_name faster_convergence
  ```
#### 问题 3：性能不足
专家配置修改（config/custom_config.yaml）：
```bash
model:
  num_experts: 7        # 专家数量从5增加
  expert_hidden_dim: 768 # 隐藏层从512增大
  experts:
    action_expert:
      hidden_dims: [768, 512, 256]  # 增加网络层数
      use_attention: true
      gating:
        top_k: 3  # 选择专家数从2增加
  temperature: 0.5      # 降低温度使选择更确定
```

#### 高级调参技巧
python
运行
```bash
# 超参数搜索脚本（scripts/hyperparameter_search.py）
import itertools

# 定义参数搜索空间
param_grid = {
    'lr': [1e-4, 5e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'dropout': [0.1, 0.2, 0.3],
    'num_experts': [5, 7],
    'top_k': [2, 3]
}

# 执行网格搜索
for params in itertools.product(*param_grid.values()):
    lr, bs, dropout, n_exp, topk = params
    exp_name = f"search_lr{lr}_bs{bs}_drop{dropout}_exp{n_exp}_k{topk}"        
    os.system(f"""
    python scripts/train_e2e.py \
        --data_dir ./data \
        --lr {lr} \
        --batch_size {bs} \
        --epochs 50 \
        --exp_name {exp_name}
    """)
```
## 3. 实时监控方法
### 3.1 终端实时监控
```bash
# 终端1：启动训练
python scripts/train_e2e.py --data_dir ./data --exp_name my_exp

# 终端2：监控日志
tail -f experiments/my_exp/logs/*.log
```
### 3.2 图形化监控
```bash
# 启动实时图表监控
python scripts/monitor_training.py \
    --exp_dir experiments/my_exp \    
    --interval 10  # 刷新间隔（秒）
```
实时图表内容：
动态 Loss 曲线
准确率变化曲线
F1 分数趋势
训练状态指标

### 3.3 TensorBoard 监控
```bash
# 启动TensorBoard服务
tensorboard --logdir experiments/

# 浏览器访问地址
http://localhost:6006
```
### 3.4 Weights & Biases 监控 Biases 监控
若使用--use_wandb参数，可在https://wandb.ai查看：
实时 Loss 曲线
学习率变化
模型架构图
超参数对比
系统资源占用
## 4. 基线模型对比
### 4.1 对比方案
在相同数据集上训练以下模型：
MoE-KGC（本模型）
GAT (图注意力网络)
GIN (图同构网络)
GraphSAGE
R-GCN (关系图卷积网络)
### 4.2 评估指标
指标	说明	理想值
Accuracy	准确率	越高越好
Precision	精确率	越高越好
Recall	召回率	越高越好
F1 Score	综合性能指标	越高越好
AUC	ROC 曲线下面积	越高越好
MRR	平均倒数排名	越高越好
Hits@1	top-1 命中率	越高越好
Hits@3	top-3 命中率	越高越好
Hits@10	top-10 命中率	越高越好
Inference Time	推理时间	越低越好
Model Size	模型参数量	适中最佳
### 4.3 结果查看
python
运行
```bash
# 读取对比结果
import pandas as pd
df = pd.read_csv('experiments/my_exp/baseline_comparison.csv')
print(df)

# 输出示例：
#        model  accuracy  precision  recall    f1    auc  inference_time  num_parameters
# 0   MoE-KGC     0.867      0.871   0.863  0.867  0.912           0.023       2,456,789
# 1       GAT     0.812      0.815   0.809  0.812  0.856           0.015       1,234,567
# 2       GIN     0.798      0.802   0.794  0.798  0.842           0.012         987,654
# 3 GraphSAGE     0.805      0.808   0.802  0.805  0.849           0.014       1,123,456
# 4     RGCN      0.821      0.824   0.818  0.821  0.863           0.018       1,567,890
```
## 5. 完整工作流
```bash
# 1. 环境准备
conda create -n moe_kgc python=3.9
conda activate moe_kgc
pip install -r requirements.txt

# 2. 数据处理
python scripts/prepare_data.py \
    --raw_data ./raw_data \
    --output_dir ./data \
    --validate --preprocess

# 3. 快速测试（10轮）
python scripts/train_e2e.py \
    --data_dir ./data \
    --epochs 10 \
    --exp_name quick_test

# 4. 查看快速测试结果
cat experiments/quick_test/final_results.json

# 5. 完整训练（100轮）
python scripts/train_e2e.py \
    --data_dir ./data \
    --epochs 100 \
    --use_wandb \
    --exp_name full_training

# 6. 实时监控
python scripts/monitor_training.py \
    --exp_dir experiments/full_training

# 7. 训练后评估
python scripts/train_e2e.py \
    --data_dir ./data \
    --eval_only \
    --checkpoint experiments/full_training/checkpoints/best_model.pt \
    --exp_name final_evaluation

# 8. 生成报告
python scripts/generate_report.py \
    --exp_dir experiments/full_training \
    --output report.pdf
```
## 6. 故障排除
常见问题解决方案
### 6.1 CUDA 内存不足
```bash
# 减小批次大小
--batch_size 16 或 8
```
### 6.2 数据格式错误
```bash
# 单独验证数据
python scripts/prepare_data.py --raw_data ./raw_data --validate
```
### 6.3 训练不收敛
```bash
# 降低学习率
--lr 0.00001
```
### 6.4 详细日志配置
python
运行
```bash
# 修改日志级别（utils/logger.py）
setup_logger(name='moe_kgc', level=logging.DEBUG)
```
## 7. 最佳实践建议
先小后大：先用 10 轮测试，再进行完整训练
检查点管理：每 10 轮自动保存，支持断点续训
实验跟踪：使用 wandb 对比多组实验结果
定期评估：通过验证集性能及时调整参数