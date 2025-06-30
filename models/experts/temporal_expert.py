import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .base_expert import BaseExpert

class TemporalExpert(BaseExpert):
    """Expert for temporal reasoning and sequence modeling"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 num_temporal_relations: int = 8,
                 max_sequence_length: int = 100):
        super().__init__(input_dim, hidden_dims, output_dim, dropout_rate, use_attention)
        
        self.num_temporal_relations = num_temporal_relations
        self.max_sequence_length = max_sequence_length
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True
        )
        
        # Transformer for long-range dependencies
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dims[0] * 2,  # Bidirectional LSTM
                nhead=8,
                dim_feedforward=hidden_dims[0] * 4,
                dropout=dropout_rate
            ),
            num_layers=2
        )
        
        # Temporal relation classifier
        self.temporal_classifier = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], num_temporal_relations)
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dims[0] // 4),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 4, hidden_dims[0] // 4)
        )
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding()

        self.feature_proj = None  # 动态创建投影层
        
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(self.max_sequence_length, self.hidden_dims[0] * 2)
        position = torch.arange(0, self.max_sequence_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.hidden_dims[0] * 2, 2).float() * 
                           -(torch.log(torch.tensor(10000.0)) / (self.hidden_dims[0] * 2)))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def compute_expert_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute temporal features"""
        batch_size = x.size(0)
        
        # Add time embeddings if timestamps provided
        if 'timestamps' in kwargs:
            timestamps = kwargs['timestamps']
            time_features = self.time_embedding(timestamps.unsqueeze(-1))
            
            # Concatenate time features with input
            if len(x.shape) == 3:
                x = torch.cat([x, time_features], dim=-1)
            else:
                x = torch.cat([x, time_features.squeeze(1)], dim=-1)
        
        # Process sequence with LSTM
        if len(x.shape) == 3:
            seq_len = x.size(1)
            
            # LSTM encoding
            lstm_out, (hidden, cell) = self.lstm(x)
            
            # Add positional encoding
            if seq_len <= self.max_sequence_length:
                pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
                lstm_out = lstm_out + pos_enc
            
            # Transformer encoding
            lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch, features]
            transformer_out = self.transformer(lstm_out)
            transformer_out = transformer_out.transpose(0, 1)  # [batch, seq_len, features]
            combined_features = transformer_out
        else:
            # For non-sequential input, expand and process
            x_expanded = x.unsqueeze(1)
            lstm_out, _ = self.lstm(x_expanded)
            combined_features = lstm_out.squeeze(1)

        # === 新增：投影到 input_dim ===
        last_dim = combined_features.shape[-1]
        if last_dim != self.input_dim:
            if (self.feature_proj is None or 
                self.feature_proj.in_features != last_dim or 
                self.feature_proj.out_features != self.input_dim):
                self.feature_proj = nn.Linear(last_dim, self.input_dim).to(combined_features.device)
            combined_features = self.feature_proj(combined_features)
        return combined_features
    
    def predict_temporal_relation(self, seq1: torch.Tensor, seq2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict temporal relation between two sequences"""
        # Encode both sequences
        enc1 = self.forward(seq1)
        enc2 = self.forward(seq2)
        
        # Pool over time dimension if needed
        if len(enc1.shape) == 3:
            enc1 = enc1.mean(dim=1)
        if len(enc2.shape) == 3:
            enc2 = enc2.mean(dim=1)
        
        # Concatenate and classify
        combined = torch.cat([enc1, enc2], dim=-1)
        relation_logits = self.temporal_classifier(combined)
        relation_probs = torch.softmax(relation_logits, dim=-1)
        
        return {
            'logits': relation_logits,
            'probabilities': relation_probs,
            'predicted_relation': torch.argmax(relation_probs, dim=-1)
        }
    
    def compute_temporal_distance(self, event1: torch.Tensor, event2: torch.Tensor) -> torch.Tensor:
        """Compute temporal distance between events"""
        enc1 = self.forward(event1)
        enc2 = self.forward(event2)
        
        # Use L2 distance in embedding space
        distance = torch.norm(enc1 - enc2, p=2, dim=-1)
        
        return distance
    
    def generate_temporal_mask(self, sequence_lengths: List[int]) -> torch.Tensor:
        """Generate attention mask for variable length sequences"""
        batch_size = len(sequence_lengths)
        max_len = max(sequence_lengths)
        
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, length in enumerate(sequence_lengths):
            mask[i, length:] = True
        
        return mask