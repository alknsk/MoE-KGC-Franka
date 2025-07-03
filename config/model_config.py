import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ExpertConfig:
    """单个专家模块的配置"""
    hidden_dims: list = field(default_factory=lambda: [512, 256])  # 专家网络的隐藏层维度
    use_attention: bool = True  # 是否在专家中使用注意力机制
    dropout_rate: float = 0.1   # dropout概率，防止过拟合

@dataclass
class GatingConfig:
    """门控机制的配置"""
    hidden_dim: int = 256
    temperature: float = 1.0  # softmax温度参数，影响门控分布的平滑程度
    noise_std: float = 0.1    # 门控噪声标准差，用于增加探索性
    top_k: int = 2            # 每次选择前k个专家
    load_balancing_weight: float = 0.01  # 负载均衡损失的权重

@dataclass
class GraphConfig:
    """图神经网络层的配置"""
    num_layers: int = 3           # 图神经网络的层数
    aggregation: str = "mean"     # 聚合方式，如mean/sum/max
    use_edge_features: bool = True  # 是否使用边特征
    edge_hidden_dim: int = 128    # 边特征的隐藏层维度

@dataclass
class TrainingConfig:
    """训练过程相关配置"""
    batch_size: int = 32          # 批大小
    learning_rate: float = 1e-4   # 学习率
    weight_decay: float = 1e-5    # 权重衰减（L2正则化）
    epochs: int = 100             # 训练轮数
    gradient_clip: float = 1.0    # 梯度裁剪阈值，防止梯度爆炸
    warmup_steps: int = 1000      # 学习率预热步数
    scheduler: str = "cosine"     # 学习率调度器类型
    accumulation_steps: int = 1   # 新增这一行
    mixed_precision: bool = False
    empty_cache_freq: int = 10
    gating_loss_weight: float = 0.01

@dataclass
class DataConfig:
    """数据处理相关配置"""
    max_seq_length: int = 512     # 最大序列长度
    vocab_size: int = 30522       # 词表大小
    num_relations: int = 50       # 关系种类数
    num_entity_types: int = 20    # 实体类型数

@dataclass
class FrankaConfig:
    """Franka机器人相关配置"""
    joint_dim: int = 7            # 机械臂关节维度
    gripper_dim: int = 2          # 夹爪维度
    force_torque_dim: int = 6     # 力/力矩传感器维度
    workspace_bounds: Dict[str, list] = field(default_factory=dict)  # 工作空间边界

@dataclass
class ModelConfig:
    """主模型配置"""
    name: str = "MoE-KGC-Franka"  # 模型名称
    hidden_dim: int = 768         # 主体隐藏层维度
    num_experts: int = 5          # 专家数量
    expert_hidden_dim: int = 512  # 专家隐藏层维度
    num_heads: int = 12           # 多头注意力头数
    dropout_rate: float = 0.1     # dropout概率
    activation: str = "gelu"      # 激活函数类型

    # 子配置
    experts: Dict[str, ExpertConfig] = field(default_factory=dict)  # 各专家配置
    gating: GatingConfig = field(default_factory=GatingConfig)      # 门控配置
    graph: GraphConfig = field(default_factory=GraphConfig)         # 图神经网络配置
    training: TrainingConfig = field(default_factory=TrainingConfig) # 训练配置
    data: DataConfig = field(default_factory=DataConfig)            # 数据配置
    franka: FrankaConfig = field(default_factory=FrankaConfig)      # Franka机器人配置

    # 路径相关
    paths: Dict[str, str] = field(default_factory=dict)             # 各类路径配置

    # 评估相关
    evaluation: Dict[str, Any] = field(default_factory=dict)        # 评估配置

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config(config_path: Optional[str] = None) -> ModelConfig:
    """
    获取配置对象
    如果未指定config_path，则默认加载当前目录下的default_config.yaml
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')

    config_dict = load_config_from_yaml(config_path)

    # 创建各专家的配置
    expert_configs = {}
    for expert_name, expert_cfg in config_dict.get('experts', {}).items():
        expert_configs[expert_name] = ExpertConfig(**expert_cfg)

    # 创建主配置对象
    model_cfg = ModelConfig(
        name=config_dict['model']['name'],
        hidden_dim=config_dict['model']['hidden_dim'],
        num_experts=config_dict['model']['num_experts'],
        expert_hidden_dim=config_dict['model']['expert_hidden_dim'],
        num_heads=config_dict['model']['num_heads'],
        dropout_rate=config_dict['model']['dropout_rate'],
        activation=config_dict['model']['activation'],
        experts=expert_configs,
        gating=GatingConfig(**config_dict['gating']),
        graph=GraphConfig(**config_dict['graph']),
        training=TrainingConfig(**config_dict['training']),
        data=DataConfig(**config_dict['data']),
        franka=FrankaConfig(**config_dict['franka']),
        paths=config_dict['paths'],
        evaluation=config_dict['evaluation']
    )

    return model_cfg

def save_config(config: ModelConfig, save_path: str):
    """将配置保存到文件"""
    config_dict = {
        'model': {
            'name': config.name,
            'hidden_dim': config.hidden_dim,
            'num_experts': config.num_experts,
            'expert_hidden_dim': config.expert_hidden_dim,
            'num_heads': config.num_heads,
            'dropout_rate': config.dropout_rate,
            'activation': config.activation
        },
        'experts': {},
        'gating': config.gating.__dict__,
        'graph': config.graph.__dict__,
        'training': config.training.__dict__,
        'data': config.data.__dict__,
        'franka': config.franka.__dict__,
        'paths': config.paths,
        'evaluation': config.evaluation
    }

    # 转换专家配置为字典
    for name, expert_cfg in config.experts.items():
        config_dict['experts'][name] = expert_cfg.__dict__

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)