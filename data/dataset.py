import torch
from torch.utils.data import Dataset
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
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
        """获取数据集长度"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """根据索引获取样本"""
        sample = self.samples[idx]
        
        # 基础图数据
        batch_data = {
            'node_features': self.features['node_features'],
            'edge_index': self.features['edge_index'],
            'edge_features': self.features['edge_features']
        }
        
        # Task-specific data
        if self.task == 'link_prediction':
            batch_data.update({
                'head': torch.tensor(sample['head'], dtype=torch.long),
                'tail': torch.tensor(sample['tail'], dtype=torch.long),
                'label': torch.tensor(sample['label'], dtype=torch.float)
            })
        
        elif self.task == 'entity_classification':
            batch_data.update({
                'node_idx': torch.tensor(sample['node_idx'], dtype=torch.long),
                'label': torch.tensor(sample['label'], dtype=torch.long)
            })
        
        elif self.task == 'relation_extraction':
            batch_data.update({
                'head_idx': torch.tensor(sample['head_idx'], dtype=torch.long),
                'tail_idx': torch.tensor(sample['tail_idx'], dtype=torch.long),
                'relation_type': torch.tensor(sample['relation_type'], dtype=torch.long)
            })
        
        return batch_data
    
    def get_collate_fn(self):
        """Get collate function for DataLoader"""
        def collate_fn(batch):
            # For simplicity, assuming single graph per batch
            # In practice, you might want to batch multiple graphs
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