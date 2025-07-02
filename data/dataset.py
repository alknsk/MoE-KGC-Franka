import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from torch_geometric.data import Data, Dataset
from torch.utils.data import Dataset
import random

"""Franka机器人知识图谱数据集"""
class FrankaKGDataset(Dataset):
    """Franka机器人知识图谱数据集"""
    
    def __init__(self, 
                 graph: nx.MultiDiGraph,
                 features: Dict[str, torch.Tensor],
                 entities: Dict[str, List],
                 relations: List[Dict],
                 config,
                 task: str = 'link_prediction'):
        """
        初始化数据集

        参数:
            graph: NetworkX图
            features: 图特征字典
            entities: 实体字典
            relations: 关系列表
            config: 配置对象
            task: 任务类型 ('link_prediction', 'entity_classification', 'relation_extraction')
        """
        self.graph = graph
        self.features = features
        self.entities = entities
        self.relations = relations
        self.config = config
        self.task = task

        # 准备特定任务的数据
        if task == 'link_prediction':
            self.prepare_link_prediction_data()
        elif task == 'entity_classification':
            self.prepare_entity_classification_data()
        elif task == 'relation_extraction':
            self.prepare_relation_extraction_data()
        else:
            raise ValueError(f"Unknown task: {task}")
    
        required_keys = [
            'node_features', 'edge_index', 'edge_features',
            'text_inputs', 'tabular_inputs', 'structured_inputs', 'node_mapping'
        ]
        for key in required_keys:
            if key not in features:
                raise KeyError(
                    f"Missing required feature '{key}' in features dict! "
                    f"Available keys: {list(features.keys())}"
                )
                
    def prepare_link_prediction_data(self):
        """准备链接预测任务的数据"""
        # 正样本（现有边）
        self.positive_samples = []
        for u, v, data in self.graph.edges(data=True):
            if u in self.features['node_mapping'] and v in self.features['node_mapping']:
                self.positive_samples.append({
                    'head': self.features['node_mapping'][u],
                    'tail': self.features['node_mapping'][v],
                    'relation': data.get('type', 'unknown'),
                    'label': 1
                })
        
        # 负样本（不存在的边）
        self.negative_samples = []
        num_negative = len(self.positive_samples)
        nodes = list(self.features['node_mapping'].values())
        
        while len(self.negative_samples) < num_negative:
            u = random.choice(nodes)
            v = random.choice(nodes)
            
            if u != v and not self.graph.has_edge(
                list(self.features['node_mapping'].keys())[u],
                list(self.features['node_mapping'].keys())[v]
            ):
                self.negative_samples.append({
                    'head': u,
                    'tail': v,
                    'relation': 'none',
                    'label': 0
                })
        
        self.samples = self.positive_samples + self.negative_samples
        random.shuffle(self.samples)
    
    def prepare_entity_classification_data(self):
        """准备实体分类任务的数据"""
        self.samples = []
        
        for node_id, node_idx in self.features['node_mapping'].items():
            node_data = self.graph.nodes[node_id]
            
            # 获取实体类型作为标签
            entity_type = node_data.get('type', 'unknown')
            type_mapping = {
                'action': 0,
                'object': 1,
                'task': 2,
                'constraint': 3,
                'safety': 4,
                'spatial': 5,
                'temporal': 6,
                'unknown': 7
            }
            
            self.samples.append({
                'node_idx': node_idx,
                'node_id': node_id,
                'label': type_mapping.get(entity_type, 7),
                'features': self.features['node_features'][node_idx]
            })
    
    def prepare_relation_extraction_data(self):
        """准备关系提取任务的数据"""
        self.samples = []
        
        # 关系类型映射
        relation_mapping = {
            'follows': 0,
            'interacts_with': 1,
            'subtask_of': 2,
            'depends_on': 3,
            'contextual': 4,
            'unknown': 5
        }
        
        for relation in self.relations:
            head = relation['head']
            tail = relation['tail']
            
            if head in self.features['node_mapping'] and tail in self.features['node_mapping']:
                self.samples.append({
                    'head_idx': self.features['node_mapping'][head],
                    'tail_idx': self.features['node_mapping'][tail],
                    'head_id': head,
                    'tail_id': tail,
                    'relation_type': relation_mapping.get(relation['type'], 5),
                    'context': relation.get('context', ''),
                    'attributes': relation.get('attributes', {})
                })
    
    def __len__(self) -> int:
        # 返回样本数量，而不是 1
        if self.task == 'link_prediction':
            return len(self.samples)
        elif self.task == 'entity_classification':
            return len(self.samples)
        elif self.task == 'relation_extraction':
            return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """根据索引获取样本"""
        
        batch_data = {
            'node_features': self.features['node_features'],      # [num_nodes, D]
            'edge_index': self.features['edge_index'],            # [2, num_edges]
            'edge_features': self.features['edge_features'],      # [num_edges, D]
            'text_inputs': self.features['text_inputs'],          # [num_nodes, seq_len]等
            'tabular_inputs': self.features['tabular_inputs'],    # [num_nodes, ...]
            'structured_inputs': self.features['structured_inputs'], # [num_nodes, ...]
        }
        
        
        # ----------- 新增：三模态占位 -----------
        # 假设 hidden_dim = self.config.hidden_dim，seq_len = 32，cat_dim = 4
        hidden_dim = getattr(self.config, 'hidden_dim', 768)
        seq_len = 128
        cat_dim = 4
        task_dim = 256
        constraint_dim = 256
        safety_dim = 256

        # 文本模态
        batch_data['text_inputs'] = {
            'input_ids': torch.zeros(1, seq_len, dtype=torch.long),
            'attention_mask': torch.ones(1, seq_len, dtype=torch.long)
        }
        # 表格模态 numerical_features=['joint_positions', 'gripper_state', 'force_torque']，共3个
        batch_data['tabular_inputs'] = {
            'numerical': torch.zeros(1, 3),         # 3个数值特征
            'categorical': {
                'action': torch.zeros(1, dtype=torch.long),      
                'object_id': torch.zeros(1, dtype=torch.long)
            }
        }
            
        # 结构化模态
        batch_data['structured_inputs'] = {
            'task_features': torch.zeros(1, task_dim),
            'constraint_features': torch.zeros(1, constraint_dim),
            'safety_features': torch.zeros(1, safety_dim)
        }
        
        # 任务相关字段
        if self.task == 'link_prediction':
            # 全部正负样本的 head/tail/label 索引
            sample = self.samples[idx]
            batch_data = {
                'node_features': self.features['node_features'],
                'edge_index': self.features['edge_index'],
                'edge_features': self.features['edge_features'],
                'text_inputs': self.features['text_inputs'],
                'tabular_inputs': self.features['tabular_inputs'],
                'structured_inputs': self.features['structured_inputs'],
                'head': torch.tensor([sample['head']], dtype=torch.long),
                'tail': torch.tensor([sample['tail']], dtype=torch.long),
                'label': torch.tensor([sample['label']], dtype=torch.float)
            }
        elif self.task == 'entity_classification':
            sample = self.samples[idx]
            batch_data = {
                'node_features': self.features['node_features'],
                'edge_index': self.features['edge_index'],
                'edge_features': self.features['edge_features'],
                'text_inputs': self.features['text_inputs'],
                'tabular_inputs': self.features['tabular_inputs'],
                'structured_inputs': self.features['structured_inputs'],
                'node_idx': torch.tensor([sample['node_idx']], dtype=torch.long),
                'label': torch.tensor([sample['label']], dtype=torch.long)
            }
        elif self.task == 'relation_extraction':
            sample = self.samples[idx]
            batch_data = {
                'node_features': self.features['node_features'],
                'edge_index': self.features['edge_index'],
                'edge_features': self.features['edge_features'],
                'text_inputs': self.features['text_inputs'],
                'tabular_inputs': self.features['tabular_inputs'],
                'structured_inputs': self.features['structured_inputs'],
                'head_idx': torch.tensor([sample['head_idx']], dtype=torch.long),
                'tail_idx': torch.tensor([sample['tail_idx']], dtype=torch.long),
                'relation_type': torch.tensor([sample['relation_type']], dtype=torch.long)
            }
        
        # 检查 edge_index 的有效性
        num_nodes = self.features['node_features'].shape[0]
        assert self.features['edge_index'].max() < num_nodes, "edge_index 中的节点索引超出了节点数量范围"
        
        return batch_data
    
    def get_collate_fn(self):
        def collate_fn(batch):
            return batch[0]

        return collate_fn
    
    def get_num_entities(self) -> int:
        """Get number of entities"""
        return len(self.features['node_mapping'])
    
    def get_num_relations(self) -> int:
        """Get number of relation types"""
        relation_types = set()
        for relation in self.relations:
            relation_types.add(relation['type'])
        return len(relation_types)
    
    def get_num_entity_types(self) -> int:
        """Get number of entity types"""
        return 8  # Based on type mapping
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_entities': self.get_num_entities(),
            'num_relations': self.get_num_relations(),
            'num_entity_types': self.get_num_entity_types(),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
            'density': nx.density(self.graph)
        }
        
        # Entity type distribution
        type_counts = {}
        for entity_type, entities in self.entities.items():
            type_counts[entity_type] = len(entities)
        stats['entity_distribution'] = type_counts
        
        return stats