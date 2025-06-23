import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import math


class GraphFusion(nn.Module):
    """Fusion module for combining outputs from multiple graph neural networks"""

    def __init__(self,
                 input_dims: List[int],
                 hidden_dim: int = 256,
                 output_dim: int = 512,
                 fusion_type: str = 'attention',
                 dropout: float = 0.1,
                 temperature: float = 1.0,
                 use_residual: bool = True,
                 use_layer_norm: bool = True):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        self.num_inputs = len(input_dims)
        self.temperature = temperature
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm

        # Input projections to common dimension
        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for dim in input_dims
        ])

        # Fusion-specific components
        if fusion_type == 'attention':
            # Multi-head attention fusion
            self.attention_heads = 8
            self.attention_dim = hidden_dim // self.attention_heads

            # Query, Key, Value projections for each input
            self.q_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_inputs)
            ])
            self.k_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_inputs)
            ])
            self.v_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_inputs)
            ])

            # Attention output projection
            self.attention_output = nn.Linear(hidden_dim, hidden_dim)

        elif fusion_type == 'gated':
            # Gated fusion with learned gates
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * self.num_inputs, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                ) for _ in range(self.num_inputs)
            ])

            # Highway network style transform gates
            self.transform_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid()
                ) for _ in range(self.num_inputs)
            ])

        elif fusion_type == 'bilinear':
            # Bilinear fusion layers
            self.bilinear_layers = nn.ModuleList()
            for i in range(self.num_inputs - 1):
                self.bilinear_layers.append(
                    nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
                )

            # Additional transformation after bilinear fusion
            self.bilinear_transform = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'tensor':
            # Tucker decomposition-based tensor fusion
            self.tucker_cores = nn.Parameter(
                torch.randn(hidden_dim, hidden_dim, self.num_inputs)
            )
            self.tucker_factors = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(self.num_inputs)
            ])

        elif fusion_type == 'mixture':
            # Mixture of fusion methods
            self.fusion_methods = ['mean', 'max', 'attention', 'gated']
            self.num_methods = len(self.fusion_methods)

            # Method selection network
            self.method_selector = nn.Sequential(
                nn.Linear(hidden_dim * self.num_inputs, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_methods),
                nn.Softmax(dim=-1)
            )

            # Sub-fusion modules
            self.attention_fusion_module = MultiHeadFusion(hidden_dim, heads=4)
            self.gated_fusion_module = GatedFusion(hidden_dim, self.num_inputs)

        # Output projection
        fusion_output_dim = hidden_dim * (1 if fusion_type != 'concat' else self.num_inputs)
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_output_dim, output_dim),
            nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        # Residual connection if input matches output dim
        if use_residual and all(dim == output_dim for dim in input_dims):
            self.residual_weight = nn.Parameter(torch.ones(self.num_inputs) / self.num_inputs)
        else:
            self.residual_weight = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

        # Fusion weight learning
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * self.num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_inputs),
            nn.Softmax(dim=-1)
        )

    def attention_fusion(self, projected_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Multi-head attention-based fusion"""
        batch_size = projected_inputs[0].size(0)

        # Stack inputs for batch processing
        stacked_inputs = torch.stack(projected_inputs, dim=1)  # [batch, num_inputs, hidden_dim]

        # Compute Q, K, V for each input
        queries = []
        keys = []
        values = []

        for i, (q_proj, k_proj, v_proj) in enumerate(zip(self.q_projections,
                                                         self.k_projections,
                                                         self.v_projections)):
            queries.append(q_proj(projected_inputs[i]))
            keys.append(k_proj(projected_inputs[i]))
            values.append(v_proj(projected_inputs[i]))

        # Stack and reshape for multi-head attention
        Q = torch.stack(queries, dim=1)  # [batch, num_inputs, hidden_dim]
        K = torch.stack(keys, dim=1)
        V = torch.stack(values, dim=1)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_inputs, self.attention_heads, self.attention_dim)
        K = K.view(batch_size, self.num_inputs, self.attention_heads, self.attention_dim)
        V = V.view(batch_size, self.num_inputs, self.attention_heads, self.attention_dim)

        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, heads, num_inputs, attention_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.attention_dim)
        attention_scores = attention_scores / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # [batch, heads, num_inputs, attention_dim]

        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, self.num_inputs, self.hidden_dim)

        # Aggregate across inputs
        fused = attention_output.mean(dim=1)

        # Output projection
        fused = self.attention_output(fused)

        return fused

    def gated_fusion(self, projected_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Gated fusion with highway networks"""
        concatenated = torch.cat(projected_inputs, dim=-1)

        # Compute gates for each input
        gated_outputs = []
        for i, (input_tensor, gate, transform_gate) in enumerate(
                zip(projected_inputs, self.gates, self.transform_gates)):
            # Compute gate value
            gate_value = gate(concatenated)

            # Highway network style gating
            transform_value = transform_gate(input_tensor)

            # Combine with transform gate
            gated_output = gate_value * input_tensor * transform_value + \
                           (1 - transform_value) * input_tensor

            gated_outputs.append(gated_output)

        # Weighted sum with learned fusion weights
        fusion_weights = self.fusion_weight_net(concatenated)
        fused = torch.zeros_like(gated_outputs[0])

        for i, gated_output in enumerate(gated_outputs):
            fused = fused + fusion_weights[:, i:i + 1] * gated_output

        return fused

    def bilinear_fusion(self, projected_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Bilinear fusion with residual connections"""
        # Start with first input
        fused = projected_inputs[0]

        # Sequentially apply bilinear fusion
        for i in range(1, len(projected_inputs)):
            # Bilinear interaction
            interaction = self.bilinear_layers[i - 1](fused, projected_inputs[i])

            # Add residual connection
            fused = F.relu(interaction + 0.5 * (fused + projected_inputs[i]))

        # Final transformation
        fused = self.bilinear_transform(fused)

        return fused

    def tensor_fusion(self, projected_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Tucker decomposition-based tensor fusion"""
        # Apply factor matrices
        factored_inputs = []
        for i, (input_tensor, factor) in enumerate(zip(projected_inputs, self.tucker_factors)):
            factored_inputs.append(factor(input_tensor))

        # Stack inputs
        stacked = torch.stack(factored_inputs, dim=-1)  # [batch, hidden_dim, num_inputs]

        # Apply Tucker core tensor
        fused = torch.einsum('bhi,hji->bhj', stacked, self.tucker_cores)
        fused = fused.sum(dim=-1)  # Sum over input dimension

        return fused

    def mixture_fusion(self, projected_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Mixture of different fusion methods"""
        concatenated = torch.cat(projected_inputs, dim=-1)

        # Select fusion method weights
        method_weights = self.method_selector(concatenated)

        # Apply different fusion methods
        fusion_results = []

        # Mean fusion
        mean_fused = torch.stack(projected_inputs).mean(dim=0)
        fusion_results.append(mean_fused)

        # Max fusion
        max_fused = torch.stack(projected_inputs).max(dim=0)[0]
        fusion_results.append(max_fused)

        # Attention fusion
        attention_fused = self.attention_fusion_module(projected_inputs)
        fusion_results.append(attention_fused)

        # Gated fusion
        gated_fused = self.gated_fusion_module(projected_inputs)
        fusion_results.append(gated_fused)

        # Weighted combination
        fused = torch.zeros_like(fusion_results[0])
        for i, result in enumerate(fusion_results):
            fused = fused + method_weights[:, i:i + 1] * result

        return fused

    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            inputs: List of input tensors from different GNNs

        Returns:
            Dictionary containing fused output and intermediate results
        """
        assert len(inputs) == self.num_inputs, \
            f"Expected {self.num_inputs} inputs, got {len(inputs)}"

        # Project inputs to common dimension
        projected_inputs = [
            proj(input_tensor) for proj, input_tensor in zip(self.input_projections, inputs)
        ]

        # Apply fusion strategy
        if self.fusion_type == 'attention':
            fused = self.attention_fusion(projected_inputs)
        elif self.fusion_type == 'gated':
            fused = self.gated_fusion(projected_inputs)
        elif self.fusion_type == 'bilinear':
            fused = self.bilinear_fusion(projected_inputs)
        elif self.fusion_type == 'tensor':
            fused = self.tensor_fusion(projected_inputs)
        elif self.fusion_type == 'mixture':
            fused = self.mixture_fusion(projected_inputs)
        elif self.fusion_type == 'concat':
            fused = torch.cat(projected_inputs, dim=-1)
        elif self.fusion_type == 'mean':
            fused = torch.stack(projected_inputs).mean(dim=0)
        elif self.fusion_type == 'max':
            fused = torch.stack(projected_inputs).max(dim=0)[0]
        elif self.fusion_type == 'sum':
            fused = torch.stack(projected_inputs).sum(dim=0)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        # Output projection
        output = self.output_projection(fused)

        # Add residual connection if applicable
        if self.residual_weight is not None:
            residual = torch.zeros_like(output)
            for i, (input_tensor, weight) in enumerate(zip(inputs, self.residual_weight)):
                residual = residual + weight * input_tensor
            output = output + residual

        # Final layer norm
        output = self.layer_norm(output)

        # Compute fusion weights for analysis
        concatenated = torch.cat(projected_inputs, dim=-1)
        fusion_weights = self.fusion_weight_net(concatenated)

        return {
            'fused_output': output,
            'projected_inputs': projected_inputs,
            'fusion_weights': fusion_weights,
            'fusion_scores': self._compute_fusion_scores(projected_inputs)
        }

    def _compute_fusion_scores(self, projected_inputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute fusion contribution scores for each input"""
        # Compute similarity between each input and the mean
        mean_representation = torch.stack(projected_inputs).mean(dim=0)

        scores = []
        for input_tensor in projected_inputs:
            similarity = F.cosine_similarity(input_tensor, mean_representation, dim=-1)
            scores.append(similarity.mean())

        return torch.stack(scores)

    def compute_fusion_entropy(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Compute entropy of fusion weights"""
        projected_inputs = [
            proj(input_tensor) for proj, input_tensor in zip(self.input_projections, inputs)
        ]
        concatenated = torch.cat(projected_inputs, dim=-1)
        fusion_weights = self.fusion_weight_net(concatenated)

        # Compute entropy
        entropy = -(fusion_weights * torch.log(fusion_weights + 1e-8)).sum(dim=-1)

        return entropy.mean()

    def get_fusion_statistics(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get detailed fusion statistics"""
        with torch.no_grad():
            output_dict = self.forward(inputs)

            stats = {
                'fusion_weights_mean': output_dict['fusion_weights'].mean(dim=0),
                'fusion_weights_std': output_dict['fusion_weights'].std(dim=0),
                'fusion_entropy': self.compute_fusion_entropy(inputs),
                'fusion_scores': output_dict['fusion_scores'],
                'output_norm': output_dict['fused_output'].norm(dim=-1).mean()
            }

            # Add fusion-specific statistics
            if self.fusion_type == 'attention' and hasattr(self, 'attention_weights'):
                stats['attention_entropy'] = -(self.attention_weights *
                                               torch.log(self.attention_weights + 1e-8)).sum(dim=-1).mean()

            return stats


class MultiHeadFusion(nn.Module):
    """Multi-head fusion module for mixture fusion"""

    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads

        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # Stack inputs
        stacked = torch.stack(inputs, dim=1)  # [batch, num_inputs, hidden_dim]

        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)

        # Average pooling
        return attended.mean(dim=1)


class GatedFusion(nn.Module):
    """Gated fusion module for mixture fusion"""

    def __init__(self, hidden_dim: int, num_inputs: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs

        self.gates = nn.Sequential(
            nn.Linear(hidden_dim * num_inputs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_inputs),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        concatenated = torch.cat(inputs, dim=-1)
        weights = self.gates(concatenated)

        fused = torch.zeros_like(inputs[0])
        for i, input_tensor in enumerate(inputs):
            fused = fused + weights[:, i:i + 1] * input_tensor

        return fused