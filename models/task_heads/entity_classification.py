import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

class EntityClassificationHead(nn.Module):
    """Task head for entity classification in knowledge graphs"""
    
    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 pooling: str = 'mean'):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.pooling = pooling
        
        # Build classification network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention mechanism for weighted pooling
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.Tanh(),
                nn.Linear(prev_dim // 2, 1)
            )
        
        # Classification head
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # Auxiliary heads for multi-task learning
        self.type_classifier = nn.Linear(prev_dim, 8)  # Entity type classes
        self.confidence_estimator = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def pool_features(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool features based on pooling strategy"""
        if len(features.shape) == 2:
            return features
        
        if self.pooling == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(features)
                pooled = (features * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = features.mean(dim=1)
        elif self.pooling == 'max':
            if mask is not None:
                features = features.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = features.max(dim=1)[0]
        elif self.pooling == 'attention' and self.use_attention:
            attention_scores = self.attention(features)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled = (features * attention_weights).sum(dim=1)
        else:
            pooled = features.mean(dim=1)
        
        return pooled
    
    def forward(self, entity_embeddings: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_confidence: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for entity classification
        
        Args:
            entity_embeddings: Entity embeddings [batch_size, (seq_len), input_dim]
            mask: Optional attention mask [batch_size, seq_len]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing logits, probabilities, and optionally confidence
        """
        # Extract features
        features = self.feature_extractor(entity_embeddings)
        
        # Pool features if needed
        pooled_features = self.pool_features(features, mask)
        
        # Main classification
        logits = self.classifier(pooled_features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Entity type classification
        type_logits = self.type_classifier(pooled_features)
        type_probs = F.softmax(type_logits, dim=-1)
        
        output = {
            'logits': logits,
            'probabilities': probabilities,
            'predicted_class': torch.argmax(probabilities, dim=-1),
            'type_logits': type_logits,
            'type_probabilities': type_probs
        }
        
        # Confidence estimation
        if return_confidence:
            confidence = self.confidence_estimator(pooled_features)
            output['confidence'] = confidence.squeeze(-1)
            
            # Compute prediction uncertainty
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=-1)
            output['uncertainty'] = entropy
        
        return output
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                    type_logits: Optional[torch.Tensor] = None,
                    type_labels: Optional[torch.Tensor] = None,
                    class_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute classification loss"""
        # Main classification loss
        if class_weights is not None:
            main_loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            main_loss = F.cross_entropy(logits, labels)
        
        losses = {'main_loss': main_loss}
        
        # Entity type loss
        if type_logits is not None and type_labels is not None:
            type_loss = F.cross_entropy(type_logits, type_labels)
            losses['type_loss'] = type_loss
            losses['total_loss'] = main_loss + 0.5 * type_loss
        else:
            losses['total_loss'] = main_loss
        
        return losses
    
    def get_embeddings(self, entity_embeddings: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get intermediate embeddings for analysis"""
        features = self.feature_extractor(entity_embeddings)
        pooled_features = self.pool_features(features, mask)
        return pooled_features