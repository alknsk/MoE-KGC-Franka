import torch
import torch.nn as nn
from typing import Dict, Optional, List
from .base_expert import BaseExpert

class ActionExpert(BaseExpert):
    """Expert for action recognition and prediction"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 num_joints: int = 7,
                 action_vocab_size: int = 50):
        super().__init__(input_dim, hidden_dims, output_dim, dropout_rate, use_attention)
        
        self.num_joints = num_joints
        self.action_vocab_size = action_vocab_size
        
        # Joint position encoder
        self.joint_encoder = nn.Sequential(
            nn.Linear(num_joints, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2)
        )
        
        # Velocity encoder (for dynamic actions)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(num_joints, hidden_dims[0] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 2, hidden_dims[0] // 2)
        )
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_vocab_size, hidden_dims[0] // 2)
        
        # Temporal convolution for action sequences
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dims[0],
            out_channels=hidden_dims[0],
            kernel_size=3,
            padding=1
        )
        
        # Action classifier head
        self.action_classifier = nn.Linear(output_dim, action_vocab_size)
        
        self.feature_proj = None  # 动态创建投影层
    
    def compute_expert_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute action-specific features
        
        Args:
            x: Input features
            kwargs: May contain 'joint_positions', 'joint_velocities', 'action_ids'
        """
        features = []
        
        # Process joint positions if available
        if 'joint_positions' in kwargs:
            joint_pos = kwargs['joint_positions']
            joint_features = self.joint_encoder(joint_pos)
            features.append(joint_features)
        
        # Process joint velocities if available
        if 'joint_velocities' in kwargs:
            joint_vel = kwargs['joint_velocities']
            velocity_features = self.velocity_encoder(joint_vel)
            features.append(velocity_features)
        
        # Process action IDs if available
        if 'action_ids' in kwargs:
            action_ids = kwargs['action_ids']
            action_features = self.action_embedding(action_ids)
            features.append(action_features)
        
        # If no specific features, use input directly
        if not features:
            return x
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)
        
        # Apply temporal convolution if sequential
        if len(combined_features.shape) == 3:
            # Reshape for conv1d: [batch, channels, length]
            combined_features = combined_features.transpose(1, 2)
            combined_features = self.temporal_conv(combined_features)
            combined_features = combined_features.transpose(1, 2)
        
        # 投影到 input_dim
        if combined_features.shape[-1] != self.input_dim:
            if (self.feature_proj is None or 
                self.feature_proj.in_features != combined_features.shape[-1] or 
                self.feature_proj.out_features != self.input_dim):
                self.feature_proj = nn.Linear(combined_features.shape[-1], self.input_dim).to(combined_features.device)
            combined_features = self.feature_proj(combined_features)
        
        return combined_features
    
    def predict_action(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Predict next action"""
        output = self.forward(x, **kwargs)
        action_logits = self.action_classifier(output)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        return {
            'logits': action_logits,
            'probabilities': action_probs,
            'predicted_action': torch.argmax(action_probs, dim=-1)
        }
    
    def compute_action_similarity(self, action1: torch.Tensor, action2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two actions"""
        # Encode both actions
        enc1 = self.forward(action1)
        enc2 = self.forward(action2)
        
        # Compute cosine similarity
        similarity = torch.cosine_similarity(enc1, enc2, dim=-1)
        
        return similarity