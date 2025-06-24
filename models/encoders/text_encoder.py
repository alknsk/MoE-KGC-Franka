import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional

class TextEncoder(nn.Module):
    """Encoder for text data from PDF documents"""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 hidden_dim: int = 768,
                 output_dim: int = 512,
                 dropout_rate: float = 0.1,
                 freeze_bert: bool = False):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            
        Returns:
            Encoded text representation [batch_size, output_dim]
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get all token embeddings
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        # Attention-based pooling
        attention_scores = self.attention(sequence_output)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.masked_fill(
            ~attention_mask.unsqueeze(-1).bool(), 
            float('-inf')
        )
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        pooled_output = torch.sum(sequence_output * attention_weights, dim=1)
        
        # Project to output dimension
        encoded = self.projection(pooled_output)
        
        return encoded
    
    def encode_text(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode raw text string"""
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Encode
        with torch.no_grad():
            output = self.forward(
                encoded['input_ids'],
                encoded['attention_mask'],
                encoded.get('token_type_ids', None)
            )
        
        return output