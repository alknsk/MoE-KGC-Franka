import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

class RelationExtractionHead(nn.Module):
    """Task head for relation extraction between entities"""
    
    def __init__(self,
                 entity_dim: int,
                 num_relations: int,
                 hidden_dim: int = 512,
                 context_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_context: bool = True,
                 use_distance_features: bool = True):
        super().__init__()
        
        self.entity_dim = entity_dim
        self.num_relations = num_relations
        self.use_context = use_context and context_dim is not None
        self.use_distance_features = use_distance_features
        
        # Calculate input dimension
        input_dim = entity_dim * 2  # Head and tail entities
        if self.use_context:
            input_dim += context_dim
        if use_distance_features:
            input_dim += 10  # Additional distance features
        
        # Feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Relation classifier
        self.relation_classifier = nn.Linear(hidden_dim // 2, num_relations)
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Context encoder
        if self.use_context:
            self.context_encoder = nn.LSTM(
                input_size=context_dim,
                hidden_size=context_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
        
        # Distance feature encoder
        if use_distance_features:
            self.distance_encoder = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
    
    def compute_distance_features(self, head_pos: torch.Tensor, 
                                tail_pos: torch.Tensor) -> torch.Tensor:
        """Compute distance-based features between entities"""
        # Euclidean distance
        euclidean_dist = torch.norm(head_pos - tail_pos, p=2, dim=-1, keepdim=True)
        
        # Manhattan distance
        manhattan_dist = torch.norm(head_pos - tail_pos, p=1, dim=-1, keepdim=True)
        
        # Directional features
        direction = (tail_pos - head_pos) / (euclidean_dist + 1e-8)
        
        # Relative position features
        rel_pos = tail_pos - head_pos
        
        # Combine features
        distance_features = torch.cat([
            euclidean_dist,
            manhattan_dist,
            direction,
            rel_pos
        ], dim=-1)
        
        # Ensure we have exactly 10 features
        if distance_features.size(-1) < 10:
            padding = torch.zeros(*distance_features.shape[:-1], 
                                10 - distance_features.size(-1),
                                device=distance_features.device)
            distance_features = torch.cat([distance_features, padding], dim=-1)
        elif distance_features.size(-1) > 10:
            distance_features = distance_features[..., :10]
        
        return distance_features
    
    def forward(self, head_embeddings: torch.Tensor, tail_embeddings: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                head_positions: Optional[torch.Tensor] = None,
                tail_positions: Optional[torch.Tensor] = None,
                return_confidence: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass for relation extraction
        
        Args:
            head_embeddings: Head entity embeddings [batch_size, entity_dim]
            tail_embeddings: Tail entity embeddings [batch_size, entity_dim]
            context: Optional context embeddings [batch_size, seq_len, context_dim]
            head_positions: Optional head entity positions [batch_size, 3]
            tail_positions: Optional tail entity positions [batch_size, 3]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing relation predictions
        """
        # Concatenate entity embeddings
        features = torch.cat([head_embeddings, tail_embeddings], dim=-1)
        
        # Add context features if available
        if self.use_context and context is not None:
            # Encode context with LSTM
            context_encoded, _ = self.context_encoder(context)
            # Pool context
            context_pooled = context_encoded.mean(dim=1)
            features = torch.cat([features, context_pooled], dim=-1)
        
        # Add distance features if available
        if self.use_distance_features and head_positions is not None and tail_positions is not None:
            distance_features = self.compute_distance_features(head_positions, tail_positions)
            distance_features = self.distance_encoder(distance_features)
            features = torch.cat([features, distance_features], dim=-1)
        
        # Extract features
        extracted_features = self.feature_extractor(features)
        
        # Classify relation
        relation_logits = self.relation_classifier(extracted_features)
        relation_probs = F.softmax(relation_logits, dim=-1)
        
        output = {
            'logits': relation_logits,
            'probabilities': relation_probs,
            'predicted_relation': torch.argmax(relation_probs, dim=-1)
        }
        
        # Estimate confidence
        if return_confidence:
            confidence = self.confidence_estimator(extracted_features)
            output['confidence'] = confidence.squeeze(-1)
            
            # Compute uncertainty
            entropy = -(relation_probs * torch.log(relation_probs + 1e-8)).sum(dim=-1)
            output['uncertainty'] = entropy
        
        return output
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                    class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute relation extraction loss"""
        if class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def get_relation_embeddings(self) -> torch.Tensor:
        """Get learned relation embeddings from classifier weights"""
        return self.relation_classifier.weight.data