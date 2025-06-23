import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

class TabularEncoder(nn.Module):
    """Encoder for tabular data from CSV files"""
    
    def __init__(self,
                 numerical_features: List[str],
                 categorical_features: List[str],
                 embedding_dims: Dict[str, int],
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 512,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleDict()
        for feat in categorical_features:
            if feat in embedding_dims:
                self.embeddings[feat] = nn.Embedding(
                    embedding_dims[feat]['vocab_size'],
                    embedding_dims[feat]['embed_dim']
                )
        
        # Calculate input dimension
        input_dim = len(numerical_features)
        for feat in categorical_features:
            if feat in embedding_dims:
                input_dim += embedding_dims[feat]['embed_dim']
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Feature-wise attention
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, 
                numerical_data: torch.Tensor,
                categorical_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            numerical_data: Numerical features [batch_size, num_numerical_features]
            categorical_data: Dict of categorical features
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        features = [numerical_data]
        
        # Embed categorical features
        for feat in self.categorical_features:
            if feat in categorical_data and feat in self.embeddings:
                embedded = self.embeddings[feat](categorical_data[feat])
                features.append(embedded)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)
        
        # Apply feature attention
        attention_weights = self.feature_attention(combined_features)
        attended_features = combined_features * attention_weights
        
        # Pass through MLP
        encoded = self.mlp(attended_features)
        
        return encoded
    
    def preprocess_franka_data(self, franka_data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Preprocess Franka robot data"""
        processed = {}
        
        # Joint positions (7 DOF)
        if 'joint_positions' in franka_data:
            joint_pos = torch.tensor(franka_data['joint_positions'], dtype=torch.float32)
            # Normalize to [-1, 1] based on joint limits
            joint_limits = torch.tensor([
                [-2.8973, 2.8973],  # Joint 1
                [-1.7628, 1.7628],  # Joint 2
                [-2.8973, 2.8973],  # Joint 3
                [-3.0718, -0.0698], # Joint 4
                [-2.8973, 2.8973],  # Joint 5
                [-0.0175, 3.7525],  # Joint 6
                [-2.8973, 2.8973]   # Joint 7
            ])
            joint_pos_norm = 2 * (joint_pos - joint_limits[:, 0]) / (joint_limits[:, 1] - joint_limits[:, 0]) - 1
            processed['joint_positions'] = joint_pos_norm
        
        # Gripper state
        if 'gripper_state' in franka_data:
            processed['gripper_state'] = torch.tensor(franka_data['gripper_state'], dtype=torch.float32)
        
        # Force/torque data
        if 'force_torque' in franka_data:
            ft_data = torch.tensor(franka_data['force_torque'], dtype=torch.float32)
            # Normalize force (N) and torque (Nm)
            ft_data[:3] = ft_data[:3] / 100.0  # Force normalization
            ft_data[3:] = ft_data[3:] / 10.0   # Torque normalization
            processed['force_torque'] = ft_data
        
        return processed