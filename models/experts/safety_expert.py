import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from .base_expert import BaseExpert

class SafetyExpert(BaseExpert):
    """Expert for safety assessment and constraint satisfaction"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 num_safety_levels: int = 5,
                 constraint_dim: int = 64):
        super().__init__(input_dim, hidden_dims, output_dim, dropout_rate, use_attention)
        
        self.num_safety_levels = num_safety_levels
        self.constraint_dim = constraint_dim
        
        # Force/torque encoder
        self.force_torque_encoder = nn.Sequential(
            nn.Linear(6, hidden_dims[0] // 3),  # 3 forces + 3 torques
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 3, hidden_dims[0] // 3)
        )
        
        # Joint limit encoder
        self.joint_limit_encoder = nn.Sequential(
            nn.Linear(14, hidden_dims[0] // 3),  # 7 joints * 2 (min/max)
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 3, hidden_dims[0] // 3)
        )
        
        # Collision detector
        self.collision_detector = nn.Sequential(
            nn.Linear(output_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        # Safety level classifier
        self.safety_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], num_safety_levels)
        )
        
        # Constraint satisfaction network
        self.constraint_network = nn.Sequential(
            nn.Linear(output_dim + constraint_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
        
        # Risk assessment module
        self.risk_assessor = nn.GRU(
            input_size=output_dim,
            hidden_size=hidden_dims[1],
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
    
    def compute_expert_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute safety-related features"""
        features = []
        
        # Process force/torque data
        if 'force_torque' in kwargs:
            ft_data = kwargs['force_torque']
            ft_features = self.force_torque_encoder(ft_data)
            features.append(ft_features)
        
        # Process joint limits
        if 'joint_limits' in kwargs:
            limits = kwargs['joint_limits']
            limit_features = self.joint_limit_encoder(limits)
            features.append(limit_features)
        
        # Process workspace boundaries
        if 'workspace_violations' in kwargs:
            violations = kwargs['workspace_violations']
            # Encode violations as features
            violation_features = torch.zeros_like(features[0]) if features else torch.zeros(
                x.size(0), self.hidden_dims[0] // 3, device=x.device
            )
            violation_features[:, :violations.size(-1)] = violations
            features.append(violation_features)
        
        # If no specific features, use input directly
        if not features:
            return x
        
        # Combine all safety features
        combined = torch.cat(features, dim=-1)
        
        # Pad if necessary
        if combined.size(-1) < x.size(-1):
            padding = torch.zeros(
                *combined.shape[:-1], 
                x.size(-1) - combined.size(-1),
                device=combined.device
            )
            combined = torch.cat([combined, padding], dim=-1)
        
        return combined
    
    def assess_collision_risk(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Assess collision risk"""
        encoded = self.forward(state, **kwargs)
        
        # Pool if sequential
        if len(encoded.shape) == 3:
            encoded = encoded.mean(dim=1)
        
        collision_prob = self.collision_detector(encoded)
        
        return {
            'collision_probability': collision_prob,
            'is_safe': collision_prob < 0.5
        }
    
    def classify_safety_level(self, state: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Classify safety level (e.g., critical, warning, normal)"""
        encoded = self.forward(state, **kwargs)
        
        # Pool if sequential
        if len(encoded.shape) == 3:
            encoded = encoded.mean(dim=1)
        
        safety_logits = self.safety_classifier(encoded)
        safety_probs = torch.softmax(safety_logits, dim=-1)
        
        return {
            'logits': safety_logits,
            'probabilities': safety_probs,
            'safety_level': torch.argmax(safety_probs, dim=-1)
        }
    
    def check_constraint_satisfaction(self, state: torch.Tensor, 
                                    constraints: torch.Tensor, **kwargs) -> torch.Tensor:
        """Check if constraints are satisfied"""
        encoded = self.forward(state, **kwargs)
        
        # Pool if sequential
        if len(encoded.shape) == 3:
            encoded = encoded.mean(dim=1)
        
        # Combine state encoding with constraints
        combined = torch.cat([encoded, constraints], dim=-1)
        
        # Predict satisfaction probability
        satisfaction_prob = self.constraint_network(combined)
        
        return satisfaction_prob.squeeze(-1)
    
    def predict_future_risk(self, state_sequence: torch.Tensor, 
                           horizon: int = 10, **kwargs) -> Dict[str, torch.Tensor]:
        """Predict future risk over time horizon"""
        # Encode sequence
        encoded = self.forward(state_sequence, **kwargs)
        
        # Use GRU for temporal risk assessment
        risk_sequence, hidden = self.risk_assessor(encoded)
        
        # Project to risk scores
        batch_size, seq_len, _ = risk_sequence.shape
        risk_scores = self.collision_detector(risk_sequence.reshape(-1, risk_sequence.size(-1)))
        risk_scores = risk_scores.reshape(batch_size, seq_len)
        
        # Predict future risk
        future_risks = []
        h = hidden
        last_output = risk_sequence[:, -1:, :]
        
        for _ in range(horizon):
            output, h = self.risk_assessor(last_output, h)
            risk = self.collision_detector(output.squeeze(1))
            future_risks.append(risk)
            last_output = output
        
        future_risks = torch.cat(future_risks, dim=1)
        
        return {
            'current_risks': risk_scores,
            'future_risks': future_risks,
            'max_risk': torch.max(torch.cat([risk_scores, future_risks], dim=1), dim=1)[0]
        }
    
    def compute_safety_margin(self, state: torch.Tensor, 
                            safety_threshold: float = 0.8, **kwargs) -> torch.Tensor:
        """Compute safety margin (distance from unsafe states)"""
        collision_risk = self.assess_collision_risk(state, **kwargs)['collision_probability']
        safety_margin = safety_threshold - collision_risk
        
        return safety_margin