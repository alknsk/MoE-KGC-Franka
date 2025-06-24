import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

class BaseExpert(nn.Module, ABC):
    """Base class for all expert modules"""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 activation: str = 'relu'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_attention = use_attention

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())

            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output_projection = nn.Linear(prev_dim, output_dim)

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                dropout=dropout_rate
            )
            self.attention_norm = nn.LayerNorm(output_dim)

    @abstractmethod
    def compute_expert_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute expert-specific features"""
        pass

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            mask: Optional attention mask
            **kwargs: Additional expert-specific arguments

        Returns:
            Expert output [batch_size, output_dim]
        """
        # Compute expert-specific features
        expert_features = self.compute_expert_features(x, **kwargs)

        # Pass through MLP
        hidden = self.mlp(expert_features)

        # Project to output dimension
        output = self.output_projection(hidden)

        # Apply attention if enabled and input is sequential
        if self.use_attention and len(output.shape) == 3:
            # Reshape for attention: [seq_len, batch_size, dim]
            output = output.transpose(0, 1)
            attended_output, _ = self.attention(output, output, output, key_padding_mask=mask)
            attended_output = attended_output.transpose(0, 1)

            # Residual connection and normalization
            output = self.attention_norm(output.transpose(0, 1) + attended_output).transpose(0, 1)

            # Average pooling over sequence dimension
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(output)
                output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                output = output.mean(dim=1)

        return output

    def get_expert_confidence(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute expert confidence for input

        Returns:
            Confidence score [batch_size, 1]
        """
        output = self.forward(x, **kwargs)
        # Use output magnitude as confidence
        confidence = torch.norm(output, p=2, dim=-1, keepdim=True)
        return torch.sigmoid(confidence)