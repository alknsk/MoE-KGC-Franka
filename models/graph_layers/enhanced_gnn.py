import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from typing import Optional, Tuple, Dict

class EnhancedGNNLayer(MessagePassing):
    """Enhanced GNN layer with multiple aggregation schemes"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 edge_dim: Optional[int] = None,
                 heads: int = 4,
                 concat: bool = True,
                 dropout: float = 0.1,
                 aggr: str = 'add'):
        super().__init__(aggr=aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        # Multi-head attention mechanism
        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        
        # Edge feature processing
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels)
        else:
            self.lin_edge = None
        
        # Output projection
        if concat:
            self.lin_out = nn.Linear(heads * out_channels, out_channels)
        else:
            self.lin_out = nn.Linear(out_channels, out_channels)
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        """
        # Multi-head transformation
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        
        # Propagate
        out = self.propagate(edge_index, query=query, key=key, value=value, 
                           edge_attr=edge_attr, size=None)
        
        # Reshape and project
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        out = self.lin_out(out)
        
        # Residual connection
        if x.size(-1) == out.size(-1):
            out = out + x
        
        # Normalization and dropout
        out = self.norm(out)
        out = self.dropout(out)
        
        return out
    
    def message(self, query_i: torch.Tensor, key_j: torch.Tensor, 
                value_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                index: torch.Tensor = None, ptr: Optional[torch.Tensor] = None,
                size_i: Optional[int] = None) -> torch.Tensor:
        """Compute messages"""
        # Compute attention scores
        alpha = (query_i * key_j).sum(dim=-1) / (self.out_channels ** 0.5)
        
        # Include edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (query_i * edge_feat).sum(dim=-1) / (self.out_channels ** 0.5)
        
        # Softmax normalization
        alpha = F.softmax(alpha, dim=-1)
        alpha = self.dropout(alpha)
        
        # Weight values by attention
        return value_j * alpha.unsqueeze(-1)


class EnhancedGNN(nn.Module):
    """Enhanced Graph Neural Network with multiple layer types"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 3,
                 edge_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 pool_type: str = 'mean'):
        super().__init__()
        
        self.num_layers = num_layers
        self.pool_type = pool_type
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(
                EnhancedGNNLayer(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    edge_dim=edge_dim,
                    heads=4,
                    concat=True,
                    dropout=dropout
                )
            )
        
        # Output layers
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 for different pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Edge feature projection
        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, edge_dim)
        else:
            self.edge_proj = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
            batch: Batch assignment [num_nodes]
        """
        # Initial projection
        x = self.input_proj(x)
        
        # Process edge features
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)
        
        # Store intermediate representations
        layer_outputs = [x]
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
            layer_outputs.append(x)
        
        # Graph-level pooling
        if batch is not None:
            # Multiple pooling strategies
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            
            # Learnable pooling
            gate = torch.sigmoid(x)
            gated_pool = global_mean_pool(x * gate, batch)
            
            # Concatenate different pooling
            graph_repr = torch.cat([mean_pool, max_pool, gated_pool], dim=-1)
        else:
            # Single graph
            mean_pool = x.mean(dim=0, keepdim=True)
            max_pool = x.max(dim=0, keepdim=True)[0]
            gate = torch.sigmoid(x)
            gated_pool = (x * gate).mean(dim=0, keepdim=True)
            graph_repr = torch.cat([mean_pool, max_pool, gated_pool], dim=-1)
        
        # Output projection
        output = self.output_proj(graph_repr)
        
        return {
            'graph_embedding': output,
            'node_embeddings': x,
            'layer_outputs': layer_outputs,
            'pooled_features': {
                'mean': mean_pool,
                'max': max_pool,
                'gated': gated_pool
            }
        }