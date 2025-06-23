import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np

class AdaptiveGating(nn.Module):
    """Adaptive gating mechanism for MoE model"""
    
    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 hidden_dim: int = 256,
                 temperature: float = 1.0,
                 noise_std: float = 0.1,
                 top_k: int = 2,
                 load_balancing_weight: float = 0.01,
                 use_learned_temperature: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.noise_std = noise_std
        self.top_k = min(top_k, num_experts)
        self.load_balancing_weight = load_balancing_weight
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Context-aware gating
        self.context_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            dropout=0.1
        )
        
        # Learned temperature parameter
        if use_learned_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.log_temperature = np.log(temperature)
        
        # Expert importance weights
        self.expert_importance = nn.Parameter(torch.ones(num_experts))
        
        # Load balancing auxiliary network
        self.load_balance_network = nn.Linear(input_dim, num_experts)
    
    def _add_noise(self, gates: torch.Tensor, training: bool) -> torch.Tensor:
        """Add Gaussian noise for exploration during training"""
        if training and self.noise_std > 0:
            noise = torch.randn_like(gates) * self.noise_std
            gates = gates + noise
        return gates
    
    def _compute_load_balancing_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss for load balancing across experts"""
        # Compute expert usage
        expert_usage = gates.mean(dim=0)
        
        # Target uniform distribution
        uniform_target = torch.ones_like(expert_usage) / self.num_experts
        
        # KL divergence from uniform
        kl_div = F.kl_div(
            torch.log(expert_usage + 1e-8),
            uniform_target,
            reduction='batchmean'
        )
        
        return kl_div * self.load_balancing_weight
    
    def forward(self, x: torch.Tensor, 
                context: Optional[torch.Tensor] = None,
                return_all_scores: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of gating mechanism
        
        Args:
            x: Input features [batch_size, input_dim]
            context: Optional context for attention [batch_size, seq_len, input_dim]
            return_all_scores: Whether to return scores for all experts
            
        Returns:
            Dictionary containing:
                - gates: Gating weights [batch_size, top_k]
                - indices: Selected expert indices [batch_size, top_k]
                - load_balancing_loss: Auxiliary loss for load balancing
                - all_scores: Scores for all experts (if requested)
        """
        batch_size = x.size(0)
        
        # Apply context attention if provided
        if context is not None:
            # x as query, context as key and value
            x_attended, _ = self.context_attention(
                x.unsqueeze(1), 
                context.transpose(0, 1), 
                context.transpose(0, 1)
            )
            x = x + x_attended.squeeze(1)
        
        # Compute gating scores
        gate_logits = self.gate_network(x)
        
        # Apply expert importance weights
        gate_logits = gate_logits * self.expert_importance
        
        # Add noise during training
        gate_logits = self._add_noise(gate_logits, self.training)
        
        # Apply temperature
        if isinstance(self.log_temperature, nn.Parameter):
            temperature = torch.exp(self.log_temperature)
        else:
            temperature = np.exp(self.log_temperature)
        
        gate_logits = gate_logits / temperature
        
        # Compute softmax probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Renormalize top-k gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balancing_loss(gate_probs)
        
        # Prepare output
        output = {
            'gates': top_k_gates,
            'indices': top_k_indices,
            'load_balancing_loss': load_balance_loss
        }
        
        if return_all_scores:
            output['all_scores'] = gate_probs
        
        return output
    
    def compute_expert_utilization(self, batch_gates: List[torch.Tensor]) -> Dict[int, float]:
        """Compute expert utilization statistics over a batch"""
        expert_counts = torch.zeros(self.num_experts)
        
        for gates_info in batch_gates:
            indices = gates_info['indices'].flatten()
            for idx in indices:
                expert_counts[idx] += 1
        
        total_selections = sum(len(g['indices'].flatten()) for g in batch_gates)
        utilization = {
            i: (count.item() / total_selections) 
            for i, count in enumerate(expert_counts)
        }
        
        return utilization
    
    def get_expert_specialization(self, inputs: torch.Tensor, 
                                 labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Analyze which experts specialize in which types of inputs"""
        with torch.no_grad():
            gates_info = self.forward(inputs, return_all_scores=True)
            expert_probs = gates_info['all_scores']
            
            if labels is not None:
                # Compute expert specialization per class
                num_classes = labels.max().item() + 1
                specialization = torch.zeros(self.num_experts, num_classes)
                
                for cls in range(num_classes):
                    mask = labels == cls
                    if mask.any():
                        specialization[:, cls] = expert_probs[mask].mean(dim=0)
                
                return specialization
            else:
                return expert_probs