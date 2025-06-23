import torch
import torch.nn as nn
from typing import Dict, List, Any , Optional
import yaml

class StructuredEncoder(nn.Module):
    """Encoder for structured data from YAML configurations"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 512,
                 dropout_rate: float = 0.1,
                 use_graph_structure: bool = True):
        super().__init__()
        
        self.use_graph_structure = use_graph_structure
        
        # Hierarchical encoding layers
        self.hierarchy_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # Task encoding
        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # Constraint encoding
        self.constraint_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # Safety encoding
        self.safety_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # Fusion layer
        fusion_input_dim = hidden_dims[0] * 2 + hidden_dims[1] * 3  # LSTM output + 3 encoders
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], output_dim)
        )
        
        # Graph-aware attention (if using graph structure)
        if use_graph_structure:
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[1],
                num_heads=4,
                dropout=dropout_rate
            )
    
    def encode_hierarchy(self, hierarchy_data: torch.Tensor) -> torch.Tensor:
        """Encode hierarchical structure"""
        # hierarchy_data: [batch_size, seq_len, input_dim]
        lstm_out, (hidden, cell) = self.hierarchy_encoder(hierarchy_data)
        
        # Use last hidden states from both directions
        hidden = hidden.view(2, 2, -1, hidden.size(-1))  # [num_layers, num_directions, batch, hidden]
        forward_hidden = hidden[-1, 0]  # Last layer, forward direction
        backward_hidden = hidden[-1, 1]  # Last layer, backward direction
        
        return torch.cat([forward_hidden, backward_hidden], dim=-1)
    
    def forward(self,
                task_features: torch.Tensor,
                constraint_features: torch.Tensor,
                safety_features: torch.Tensor,
                hierarchy_features: Optional[torch.Tensor] = None,
                graph_adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            task_features: Task features [batch_size, input_dim]
            constraint_features: Constraint features [batch_size, input_dim]
            safety_features: Safety features [batch_size, input_dim]
            hierarchy_features: Hierarchical features [batch_size, seq_len, input_dim]
            graph_adjacency: Graph adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        # Encode different components
        task_encoded = self.task_encoder(task_features)
        constraint_encoded = self.constraint_encoder(constraint_features)
        safety_encoded = self.safety_encoder(safety_features)
        
        # Encode hierarchy if provided
        if hierarchy_features is not None:
            hierarchy_encoded = self.encode_hierarchy(hierarchy_features)
        else:
            hierarchy_encoded = torch.zeros(
                task_features.size(0), 
                self.hierarchy_encoder.hidden_size * 2,
                device=task_features.device
            )
        
        # Apply graph attention if using graph structure
        if self.use_graph_structure and graph_adjacency is not None:
            # Stack encoded features for attention
            stacked_features = torch.stack([
                task_encoded, constraint_encoded, safety_encoded
            ], dim=1)  # [batch_size, 3, hidden_dim]
            
            # Apply graph-aware attention
            attended_features, _ = self.graph_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Flatten attended features
            task_encoded = attended_features[:, 0]
            constraint_encoded = attended_features[:, 1]
            safety_encoded = attended_features[:, 2]
        
        # Fuse all encodings
        fused_features = torch.cat([
            hierarchy_encoded,
            task_encoded,
            constraint_encoded,
            safety_encoded
        ], dim=-1)
        
        # Final encoding
        encoded = self.fusion(fused_features)
        
        return encoded
    
    def parse_yaml_features(self, yaml_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Parse YAML data into feature tensors"""
        features = {}
        
        # Extract task features
        if 'tasks' in yaml_data:
            task_features = []
            for task in yaml_data['tasks']:
                # Create feature vector from task parameters
                feat = torch.zeros(256)  # Placeholder dimension
                # Add custom feature extraction logic here
                task_features.append(feat)
            features['task_features'] = torch.stack(task_features)
        
        # Extract constraint features
        if 'constraints' in yaml_data:
            constraint_features = []
            for constraint in yaml_data['constraints']:
                feat = torch.zeros(256)
                # Add custom feature extraction logic here
                constraint_features.append(feat)
            features['constraint_features'] = torch.stack(constraint_features)
        
        # Extract safety features
        if 'safety_limits' in yaml_data:
            safety_feat = torch.zeros(256)
            # Add custom feature extraction logic here
            features['safety_features'] = safety_feat.unsqueeze(0)
        
        return features