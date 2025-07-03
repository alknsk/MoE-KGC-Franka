import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
from torch_geometric.utils import negative_sampling
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

class MoEKGCBatchDataLoader:
    """
    支持mini-batch子图采样的数据加载器
    专门为MoE-KGC模型设计，处理多模态输入
    """
    def __init__(self, 
                 pyg_data: Data,
                 batch_size: int = 32,
                 num_neighbors: List[int] = [25, 10],
                 sampling_method: str = 'neighbor',
                 num_workers: int = 4,
                 shuffle: bool = True,
                 mode: str = 'train',
                 task: str = 'link_prediction'):
        
        self.pyg_data = pyg_data
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.sampling_method = sampling_method
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.mode = mode
        self.task = task
        self.logger = logging.getLogger(__name__)
        
        # 验证数据完整性
        self._validate_data()
        
    def _validate_data(self):
        """验证PyG数据对象的完整性"""
        required_attrs = ['x', 'edge_index']
        for attr in required_attrs:
            if not hasattr(self.pyg_data, attr):
                raise ValueError(f"PyG data missing required attribute: {attr}")
                
        # 验证多模态数据
        if hasattr(self.pyg_data, 'text_inputs'):
            assert isinstance(self.pyg_data.text_inputs, dict), "text_inputs must be a dict"
            assert 'input_ids' in self.pyg_data.text_inputs
            assert 'attention_mask' in self.pyg_data.text_inputs
            
    def get_link_prediction_loader(self):
        """链接预测任务的批处理加载器"""
        # 准备边标签
        if self.mode == 'train' and hasattr(self.pyg_data, 'train_edge_index'):
            edge_label_index = self.pyg_data.train_edge_index
        elif self.mode == 'val' and hasattr(self.pyg_data, 'val_edge_index'):
            edge_label_index = self.pyg_data.val_edge_index
        elif self.mode == 'test' and hasattr(self.pyg_data, 'test_edge_index'):
            edge_label_index = self.pyg_data.test_edge_index
        else:
            # 如果没有预定义的分割，使用所有边
            edge_label_index = self.pyg_data.edge_index
            
        # 创建正样本标签
        pos_edge_label = torch.ones(edge_label_index.size(1))
        
        # 负采样
        neg_edge_index = negative_sampling(
            edge_index=self.pyg_data.edge_index,
            num_nodes=self.pyg_data.num_nodes,
            num_neg_samples=edge_label_index.size(1)
        )
        neg_edge_label = torch.zeros(neg_edge_index.size(1))
        
        # 合并正负样本
        self.pyg_data.edge_label_index = torch.cat([edge_label_index, neg_edge_index], dim=1)
        self.pyg_data.edge_label = torch.cat([pos_edge_label, neg_edge_label])
        
        # 使用NeighborLoader进行子图采样
        loader = NeighborLoader(
            self.pyg_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            edge_label_index=self.pyg_data.edge_label_index,
            edge_label=self.pyg_data.edge_label,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            transform=self._transform_batch
        )
        
        return loader
    
    def get_node_classification_loader(self):
        """节点分类任务的批处理加载器"""
        # 获取有标签的节点
        if self.mode == 'train' and hasattr(self.pyg_data, 'train_mask'):
            mask = self.pyg_data.train_mask
        elif self.mode == 'val' and hasattr(self.pyg_data, 'val_mask'):
            mask = self.pyg_data.val_mask
        elif self.mode == 'test' and hasattr(self.pyg_data, 'test_mask'):
            mask = self.pyg_data.test_mask
        else:
            # 使用所有节点
            mask = torch.ones(self.pyg_data.num_nodes, dtype=torch.bool)
            
        node_indices = mask.nonzero(as_tuple=False).view(-1)
        
        # 使用NeighborLoader
        loader = NeighborLoader(
            self.pyg_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            input_nodes=node_indices,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            transform=self._transform_batch
        )
        
        return loader
    
    def _transform_batch(self, batch: Data) -> Data:
        """转换批次数据，确保格式正确"""
        # 确保必要的属性存在
        if not hasattr(batch, 'batch'):
            # 对于单图，创建batch向量
            batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long)
            
        # 处理多模态数据的批处理
        if hasattr(self.pyg_data, 'text_inputs'):
            # 提取子图对应的文本输入
            batch.text_inputs = {
                'input_ids': self.pyg_data.text_inputs['input_ids'][batch.n_id],
                'attention_mask': self.pyg_data.text_inputs['attention_mask'][batch.n_id]
            }
            
        if hasattr(self.pyg_data, 'tabular_inputs'):
            batch.tabular_inputs = {
                'numerical': self.pyg_data.tabular_inputs['numerical'][batch.n_id],
                'categorical': {}
            }
            for key, val in self.pyg_data.tabular_inputs['categorical'].items():
                batch.tabular_inputs['categorical'][key] = val[batch.n_id]
                
        if hasattr(self.pyg_data, 'structured_inputs'):
            batch.structured_inputs = {
                'task_features': self.pyg_data.structured_inputs['task_features'][batch.n_id],
                'constraint_features': self.pyg_data.structured_inputs['constraint_features'][batch.n_id],
                'safety_features': self.pyg_data.structured_inputs['safety_features'][batch.n_id]
            }
            
        # 处理节点特征
        if hasattr(self.pyg_data, 'x'):
            batch.node_features = self.pyg_data.x[batch.n_id]
        else:
            batch.node_features = torch.zeros(len(batch.n_id), 768)  # 使用默认维度
            
        # 处理任务特定的数据
        if self.task == 'link_prediction' and hasattr(batch, 'edge_label_index'):
            # 映射边索引到子图节点索引
            batch.head = batch.edge_label_index[0][batch.edge_label == 1]
            batch.tail = batch.edge_label_index[1][batch.edge_label == 1]
            batch.label = batch.edge_label[batch.edge_label == 1]
            
            # 负样本
            neg_mask = batch.edge_label == 0
            if neg_mask.any():
                batch.neg_head = batch.edge_label_index[0][neg_mask]
                batch.neg_tail = batch.edge_label_index[1][neg_mask]
                
        elif self.task == 'entity_classification':
            if hasattr(self.pyg_data, 'y'):
                batch.node_idx = torch.arange(len(batch.n_id))
                batch.label = self.pyg_data.y[batch.n_id]
                
        return batch