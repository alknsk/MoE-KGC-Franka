import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .base_expert import BaseExpert

class SemanticExpert(BaseExpert):
    """Expert for semantic understanding and reasoning"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 vocab_size: int = 30522,
                 num_semantic_categories: int = 20):
        super().__init__(input_dim, hidden_dims, output_dim, dropout_rate, use_attention)
        
        self.vocab_size = vocab_size
        self.num_semantic_categories = num_semantic_categories
        
        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, hidden_dims[0] // 2)
        
        # Concept embedding
        self.concept_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0] // 2)
        )
        
        # Semantic category classifier
        self.category_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], num_semantic_categories)
        )
        
        # Relation embedding network
        self.relation_encoder = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1])
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[0] // 2,
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Semantic similarity head
        self.similarity_head = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
    
    def compute_expert_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute semantic features"""
        features = []
        
        # Process word indices if available
        if 'word_indices' in kwargs:
            word_indices = kwargs['word_indices']
            word_features = self.word_embedding(word_indices)
            features.append(word_features)
        
        # Process concept features
        if 'concept_features' in kwargs:
            concepts = kwargs['concept_features']
            concept_features = self.concept_embedding(concepts)
            features.append(concept_features)
        
        # If no specific features, encode input as concepts
        if not features:
            concept_features = self.concept_embedding(x)
            features.append(concept_features)
        
        # Apply cross-modal attention if multiple feature types
        if len(features) > 1 and len(features[0].shape) == 3:
            # Use first feature type as query, others as key/value
            query = features[0].transpose(0, 1)
            key = torch.cat(features[1:], dim=-1).transpose(0, 1)
            value = key
            
            attended, _ = self.cross_attention(query, key, value)
            attended = attended.transpose(0, 1)
            
            return attended
        else:
            return torch.cat(features, dim=-1) if len(features) > 1 else features[0]
    
    def classify_semantic_category(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Classify input into semantic categories"""
        encoded = self.forward(x, **kwargs)
        
        # Pool if sequential
        if len(encoded.shape) == 3:
            encoded = encoded.mean(dim=1)
        
        category_logits = self.category_classifier(encoded)
        category_probs = torch.softmax(category_logits, dim=-1)
        
        return {
            'logits': category_logits,
            'probabilities': category_probs,
            'predicted_category': torch.argmax(category_probs, dim=-1)
        }
    
    def compute_semantic_similarity(self, item1: torch.Tensor, item2: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute semantic similarity between two items"""
        # Encode both items
        enc1 = self.forward(item1, **kwargs)
        enc2 = self.forward(item2, **kwargs)
        
        # Pool if needed
        if len(enc1.shape) == 3:
            enc1 = enc1.mean(dim=1)
        if len(enc2.shape) == 3:
            enc2 = enc2.mean(dim=1)
        
        # Compute similarity
        combined = torch.cat([enc1, enc2], dim=-1)
        similarity = self.similarity_head(combined)
        
        return similarity.squeeze(-1)
    
    def encode_semantic_relation(self, head: torch.Tensor, tail: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode semantic relation between entities"""
        # Encode entities
        head_enc = self.forward(head, **kwargs)
        tail_enc = self.forward(tail, **kwargs)
        
        # Pool if needed
        if len(head_enc.shape) == 3:
            head_enc = head_enc.mean(dim=1)
        if len(tail_enc.shape) == 3:
            tail_enc = tail_enc.mean(dim=1)
        
        # Encode relation
        combined = torch.cat([head_enc, tail_enc], dim=-1)
        relation_encoding = self.relation_encoder(combined)
        
        return relation_encoding
    
    def generate_concept_graph(self, concepts: List[torch.Tensor]) -> torch.Tensor:
        """Generate concept graph from list of concepts"""
        num_concepts = len(concepts)
        
        # Encode all concepts
        encoded_concepts = []
        for concept in concepts:
            enc = self.forward(concept)
            if len(enc.shape) == 3:
                enc = enc.mean(dim=1)
            encoded_concepts.append(enc)
        
        encoded_concepts = torch.stack(encoded_concepts)
        
        # Compute pairwise similarities as adjacency matrix
        adjacency = torch.zeros(num_concepts, num_concepts)
        for i in range(num_concepts):
            for j in range(i+1, num_concepts):
                sim = self.compute_semantic_similarity(
                    encoded_concepts[i].unsqueeze(0),
                    encoded_concepts[j].unsqueeze(0)
                )
                adjacency[i, j] = sim
                adjacency[j, i] = sim
        
        return adjacency