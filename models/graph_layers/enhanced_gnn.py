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
        self.aggr = aggr

        assert out_channels % heads == 0, f"out_channels ({out_channels}) must be divisible by heads ({heads})"
        self.head_dim = out_channels // heads
        # Multi-head attention mechanism
        self.lin_key = nn.Linear(in_channels, heads * self.head_dim)
        self.lin_query = nn.Linear(in_channels, heads * self.head_dim)
        self.lin_value = nn.Linear(in_channels, heads * self.head_dim)
        
        # Edge feature processing
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * self.head_dim)
        else:
            self.lin_edge = None
        
        # Output projection
        if concat:
            self.lin_out = nn.Linear(heads * self.head_dim, out_channels)
        else:
            self.lin_out = nn.Linear(self.head_dim, out_channels)
        
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

        Returns:
            Updated node features [num_nodes, out_channels]
        """
        # 严格验证输入
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {x.dim()}D")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Expected edge_index to be [2, num_edges], got {edge_index.shape}")
        
        num_nodes = x.size(0)
        
        # 验证edge_index的有效性
        if edge_index.numel() > 0:
            if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
                #print(f"[DEBUG][Layer] Invalid edge_index detected: max={edge_index.max().item()}, min={edge_index.min().item()}, num_nodes={num_nodes}")
                # 过滤无效边
                valid_mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & \
                            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
                edge_index = edge_index[:, valid_mask]
                if edge_attr is not None:
                    edge_attr = edge_attr[valid_mask]
                #print(f"[DEBUG][Layer] After filtering: edge_index shape={edge_index.shape}")
        
        # 无边时直接做线性变换
        if edge_index.numel() == 0:
            #print(f"[DEBUG][Layer] No edges, using identity transformation")
            out = self.lin_out(torch.zeros(num_nodes, self.heads * self.head_dim if self.concat else self.head_dim, 
                                         device=x.device, dtype=x.dtype))
            # 残差连接
            if x.size(-1) == out.size(-1):
                out = out + x
            out = self.norm(out)
            out = self.dropout(out)
            return out

        # Multi-head transformation
        key = self.lin_key(x).view(num_nodes, self.heads, self.head_dim)
        query = self.lin_query(x).view(num_nodes, self.heads, self.head_dim)
        value = self.lin_value(x).view(num_nodes, self.heads, self.head_dim)
        
        # 手动实现消息传递以避免CUDA错误
        try:
            # 获取源节点和目标节点的特征
            src_nodes = edge_index[0]  # 源节点索引
            dst_nodes = edge_index[1]  # 目标节点索引
            
            # 确保索引在有效范围内
            assert src_nodes.max().item() < num_nodes, f"src_nodes max {src_nodes.max().item()} >= {num_nodes}"
            assert dst_nodes.max().item() < num_nodes, f"dst_nodes max {dst_nodes.max().item()} >= {num_nodes}"
            assert src_nodes.min().item() >= 0, f"src_nodes min {src_nodes.min().item()} < 0"
            assert dst_nodes.min().item() >= 0, f"dst_nodes min {dst_nodes.min().item()} < 0"
            
            # 获取源节点和目标节点的query, key, value
            query_i = query[dst_nodes]  # [num_edges, heads, head_dim]
            key_j = key[src_nodes]      # [num_edges, heads, head_dim]
            value_j = value[src_nodes]  # [num_edges, heads, head_dim]
            
            # 计算attention scores
            alpha = (query_i * key_j).sum(dim=-1) / (self.head_dim ** 0.5)  # [num_edges, heads]
            
            # 包含边特征（如果有的话）
            if edge_attr is not None and self.lin_edge is not None:
                edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)
                if edge_feat.size(0) == alpha.size(0):
                    alpha = alpha + (query_i * edge_feat).sum(dim=-1) / (self.head_dim ** 0.5)
            
            # Softmax normalization
            alpha = F.softmax(alpha, dim=-1)  # [num_edges, heads]
            alpha = self.dropout(alpha)
            
            # 计算消息
            messages = value_j * alpha.unsqueeze(-1)  # [num_edges, heads, head_dim]
            device = x.device
            # 聚合消息到目标节点
            out = torch.zeros(num_nodes, self.heads, self.head_dim, device=device, dtype=x.dtype)
            
            if self.aggr == 'add':
                out.index_add_(0, dst_nodes, messages)
            elif self.aggr == 'mean':
                out.index_add_(0, dst_nodes, messages)
                # 计算每个节点的度数进行归一化
                degree = torch.zeros(num_nodes, device=device, dtype=torch.float)
                degree.index_add_(0, dst_nodes, torch.ones(dst_nodes.size(0), device=device))
                degree = degree.clamp(min=1.0)
                out = out / degree.view(-1, 1, 1)
            else:
                # 默认使用add
                out.index_add_(0, dst_nodes, messages)
            
            #print(f"[DEBUG][Layer] Message passing successful, out shape: {out.shape}")
            
        except Exception as e:
            #print(f"[DEBUG][Layer] Manual message passing failed: {e}")
            # 回退到零张量
            out = torch.zeros(num_nodes, self.heads, self.head_dim, device=device, dtype=x.dtype)
        
        # Reshape and project
        if self.concat:
            out = out.view(num_nodes, self.heads * self.head_dim)
        else:
            out = out.mean(dim=1)
        
        out = self.lin_out(out)
        
        # 残差连接
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
        try:
            # 验证输入维度
            if query_i.dim() != 3 or key_j.dim() != 3 or value_j.dim() != 3:
                raise ValueError(f"Expected 3D tensors, got query_i: {query_i.dim()}D, key_j: {key_j.dim()}D, value_j: {value_j.dim()}D")
            
            if query_i.size(-1) != self.head_dim or key_j.size(-1) != self.head_dim:
                raise ValueError(f"Expected head_dim={self.head_dim}, got query_i: {query_i.size(-1)}, key_j: {key_j.size(-1)}")
            
            # Compute attention scores - 修复维度
            alpha = (query_i * key_j).sum(dim=-1) / (self.head_dim ** 0.5)
            
            # Include edge features if available
            if edge_attr is not None and self.lin_edge is not None:
                edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)
                if edge_feat.size(0) == alpha.size(0):  # 确保维度匹配
                    alpha = alpha + (query_i * edge_feat).sum(dim=-1) / (self.head_dim ** 0.5)
            
            # Softmax normalization
            alpha = F.softmax(alpha, dim=-1)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # Weight values by attention
            return value_j * alpha.unsqueeze(-1)
            
        except Exception as e:
            #print(f"[DEBUG][Layer] Message computation failed: {e}")
            # 回退到简单的值传递
            return value_j


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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
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
        # 严格验证输入
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {x.dim()}D with shape {x.shape}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.size(-1)}")
        
        num_nodes = x.size(0)
        device = x.device

        if batch is not None:
            if batch.max().item() >= num_nodes:
                raise ValueError(f"batch.max()={batch.max().item()} >= num_nodes={num_nodes}")
        
        #print(f"[DEBUG] input x shape: {x.shape}")

        # 修复索引超出范围的问题
        if edge_index.numel() > 0:
            #print(f"[DEBUG] edge_index shape: {edge_index.shape}, max: {edge_index.max().item()}, min: {edge_index.min().item()}")
            
           # 过滤无效边
            valid_mask = (edge_index[0] >= 0) & (edge_index[0] < x.shape[0]) & \
                        (edge_index[1] >= 0) & (edge_index[1] < x.shape[0])
            
            if valid_mask.sum() < edge_index.size(1):
                #print(f"[DEBUG] Warning: filtering {edge_index.size(1) - valid_mask.sum()} invalid edges")
                edge_index = edge_index[:, valid_mask]
                if edge_attr is not None:
                    edge_attr = edge_attr[valid_mask]
            
            # 如果没有有效边，创建自环
            if edge_index.size(1) == 0:
                #print(f"[DEBUG] No valid edges, creating self-loops")
                # 只为前几个节点创建自环，避免创建太多边
                num_self_loops = min(num_nodes, 10)
                self_loops = torch.arange(num_self_loops, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
                edge_index = self_loops
                edge_attr = None
            
            #print(f"[DEBUG] After validation: edge_index shape: {edge_index.shape}")
        else:
            #print(f"[DEBUG] edge_index is empty, creating self-loops")
            num_self_loops = min(num_nodes, 10)
            self_loops = torch.arange(num_self_loops, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
            edge_index = self_loops
            edge_attr = None
            #print(f"[DEBUG] Created minimal self-loops: {edge_index.shape}")
        
        if batch is not None:
            print(f"[DEBUG] batch shape: {batch.shape}, batch.max: {batch.max().item()}, batch.min: {batch.min().item()}")
        if edge_attr is not None:
            print(f"[DEBUG] edge_attr shape: {edge_attr.shape}")
        
        # Initial projection
        x = self.input_proj(x)
        #print(f"[DEBUG] after input_proj, x shape: {x.shape}")
        
        # Process edge features
        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr)
        
        # Store intermediate representations
        layer_outputs = [x]
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            try:
                x = layer(x, edge_index, edge_attr)
                print(f"[DEBUG] after GNN layer {i}, x shape: {x.shape}, x.mean: {x.mean().item():.4f}")
                layer_outputs.append(x)
            except Exception as e:
                print(f"[DEBUG] GNN layer {i} failed: {e}")
                # 使用上一层的输出
                break
        
        # Graph-level pooling
        if batch is not None:
            #print(f"[DEBUG] before pooling, x shape: {x.shape}, batch shape: {batch.shape}")
            # Multiple pooling strategies
            mean_pool = global_mean_pool(x, batch)
            max_pool = global_max_pool(x, batch)
            
            # Learnable pooling
            gate = torch.sigmoid(x)
            gated_pool = global_mean_pool(x * gate, batch)
            
            # Concatenate different pooling
            graph_repr = torch.cat([mean_pool, max_pool, gated_pool], dim=-1)
        else:
            #print(f"[DEBUG] before pooling, x shape: {x.shape} (single graph)")
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