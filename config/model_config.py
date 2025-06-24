import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class ExpertConfig:
    """Configuration for individual expert modules"""
    hidden_dims: list = field(default_factory=lambda: [512, 256])
    use_attention: bool = True
    dropout_rate: float = 0.1

@dataclass
class GatingConfig:
    """Configuration for gating mechanism"""
    temperature: float = 1.0
    noise_std: float = 0.1
    top_k: int = 2
    load_balancing_weight: float = 0.01

@dataclass
class GraphConfig:
    """Configuration for graph neural network layers"""
    num_layers: int = 3
    aggregation: str = "mean"
    use_edge_features: bool = True
    edge_hidden_dim: int = 128

@dataclass
class TrainingConfig:
    """Configuration for training process"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    scheduler: str = "cosine"

@dataclass
class DataConfig:
    """Configuration for data processing"""
    max_seq_length: int = 512
    vocab_size: int = 30522
    num_relations: int = 50
    num_entity_types: int = 20

@dataclass
class FrankaConfig:
    """Configuration specific to Franka robot"""
    joint_dim: int = 7
    gripper_dim: int = 2
    force_torque_dim: int = 6
    workspace_bounds: Dict[str, list] = field(default_factory=dict)

@dataclass
class ModelConfig:
    """Main model configuration"""
    name: str = "MoE-KGC-Franka"
    hidden_dim: int = 768
    num_experts: int = 5
    expert_hidden_dim: int = 512
    num_heads: int = 12
    dropout_rate: float = 0.1
    activation: str = "gelu"

    # Sub-configurations
    experts: Dict[str, ExpertConfig] = field(default_factory=dict)
    gating: GatingConfig = field(default_factory=GatingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    franka: FrankaConfig = field(default_factory=FrankaConfig)

    # Paths
    paths: Dict[str, str] = field(default_factory=dict)

    # Evaluation
    evaluation: Dict[str, Any] = field(default_factory=dict)

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_config(config_path: Optional[str] = None) -> ModelConfig:
    """Get configuration object"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'default_config.yaml')

    config_dict = load_config_from_yaml(config_path)

    # Create sub-configurations
    expert_configs = {}
    for expert_name, expert_cfg in config_dict.get('experts', {}).items():
        expert_configs[expert_name] = ExpertConfig(**expert_cfg)

    # Create main configuration
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
    """Save configuration to file"""
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

    # Convert expert configs
    for name, expert_cfg in config.experts.items():
        config_dict['experts'][name] = expert_cfg.__dict__

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)