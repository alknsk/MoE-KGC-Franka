import os
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
import numpy as np
from pathlib import Path
import pickle
import json

from .preprocessors import PDFProcessor, CSVProcessor, YAMLProcessor
from .dataset import FrankaKGDataset

class KGDataLoader:
    """知识图谱构建的主数据加载器"""

    def __init__(self, config):
        # 保存配置对象
        self.config = config
        # 初始化各类文件处理器
        self.pdf_processor = PDFProcessor()
        self.csv_processor = CSVProcessor()
        self.yaml_processor = YAMLProcessor()
        # 存储实体、关系和图结构
        self.entities = {}
        self.relations = []
        self.graph = nx.MultiDiGraph()
        
    def load_data_from_directory(self, data_dir: str) -> Dict[str, Any]:
        """
        从指定目录加载所有原始数据文件（PDF/CSV/YAML）
        参数:
            data_dir: 数据目录路径
        返回:
            dict: 包含所有类型数据的字典
        """
        data_dir = Path(data_dir)
        all_data = {
            'pdf_data': [],
            'csv_data': [],
            'yaml_data': []
        }
        
        # 处理PDF文件
        for pdf_file in data_dir.glob("*.pdf"):
            print(f"Processing PDF: {pdf_file}")
            pdf_data = self.pdf_processor.process(str(pdf_file))
            all_data['pdf_data'].append(pdf_data)
        
        # 处理CSV文件
        for csv_file in data_dir.glob("*.csv"):
            print(f"Processing CSV: {csv_file}")
            csv_data = self.csv_processor.process(str(csv_file))
            all_data['csv_data'].append(csv_data)
        
        # 处理YAML文件
        for yaml_file in data_dir.glob("*.yaml"):
            print(f"Processing YAML: {yaml_file}")
            yaml_data = self.yaml_processor.process(str(yaml_file))
            all_data['yaml_data'].append(yaml_data)
        
        return all_data
    
    def merge_entities(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        合并不同来源的数据实体，统一格式
        参数:
            all_data: 各类型原始数据
        返回:
            dict: 合并后的实体字典，按类型分类
        """
        merged_entities = {
            'action': [],
            'object': [],
            'task': [],
            'constraint': [],
            'safety': [],
            'spatial': [],
            'temporal': []
        }
        
        entity_id_counter = 0
        
        # 合并PDF实体
        for pdf_data in all_data['pdf_data']:
            for entity_type, entity_list in pdf_data.get('entities', {}).items():
                if entity_type in merged_entities:
                    for entity in entity_list:
                        merged_entities[entity_type].append({
                            'id': f"entity_{entity_id_counter}",
                            'text': entity[0] if isinstance(entity, tuple) else entity,
                            'source': 'pdf',
                            'type': entity_type
                        })
                        entity_id_counter += 1
        
        # 合并CSV实体
        for csv_data in all_data['csv_data']:
            entities = csv_data.get('entities', {})
            for entity in entities.get('actions', []):
                merged_entities['action'].append({
                    'id': entity['id'],
                    'name': entity['name'],
                    'attributes': entity['attributes'],
                    'source': 'csv',
                    'type': 'action'
                })
            
            for entity in entities.get('objects', []):
                merged_entities['object'].append({
                    'id': entity['id'],
                    'name': entity['name'],
                    'attributes': entity.get('attributes', {}),
                    'source': 'csv',
                    'type': 'object'
                })
        
        # 合并YAML实体
        for yaml_data in all_data['yaml_data']:
            entities = yaml_data.get('entities', {})
            for entity in entities.get('tasks', []):
                merged_entities['task'].append(entity)
            
            for entity in entities.get('constraints', []):
                merged_entities['constraint'].append(entity)
            
            for entity in entities.get('safety', []):
                merged_entities['safety'].append(entity)
        
        return merged_entities
    
    def merge_relations(self, all_data: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
        """
        合并不同来源的数据关系，统一格式
        参数:
            all_data: 各类型原始数据
        返回:
            list: 合并后的关系列表
        """
        merged_relations = []
        
        # 合并PDF关系
        for pdf_data in all_data['pdf_data']:
            for relation in pdf_data.get('relations', []):
                merged_relations.append({
                    'head': relation['head']['text'],
                    'tail': relation['tail']['text'],
                    'type': 'contextual',
                    'context': relation.get('context', ''),
                    'source': 'pdf'
                })
        
        # 合并CSV关系
        for csv_data in all_data['csv_data']:
            relations = csv_data.get('relations', {})
            
            for relation in relations.get('temporal', []):
                merged_relations.append({
                    **relation,
                    'source': 'csv'
                })
            
            for relation in relations.get('interactions', []):
                merged_relations.append({
                    **relation,
                    'source': 'csv'
                })
        
        # 合并YAML关系
        for yaml_data in all_data['yaml_data']:
            relations = yaml_data.get('relations', {})
            
            for relation in relations.get('hierarchical', []):
                merged_relations.append({
                    **relation,
                    'source': 'yaml'
                })
        
        return merged_relations
    
    def build_knowledge_graph(self, entities: Dict[str, List], relations: List[Dict]) -> nx.MultiDiGraph:
        """
        根据实体和关系构建NetworkX多重有向图
        参数:
            entities: 按类型分类的实体字典
            relations: 关系列表
        返回:
            nx.MultiDiGraph: 构建的知识图谱
        """
        G = nx.MultiDiGraph()
        
        # 添加节点
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                G.add_node(
                    entity['id'],
                    **entity
                )
        
        # 添加边
        for relation in relations:
            G.add_edge(
                relation['head'],
                relation['tail'],
                **relation
            )
        
        return G
    
    def extract_graph_features(self, G: nx.MultiDiGraph) -> Dict[str, torch.Tensor]:
        """
        从知识图谱中提取特征，供模型输入
        参数:
            G: NetworkX图
        返回:
            dict: 包含节点特征、边索引、边特征等
        """
        # 节点特征
        node_features = []
        node_mapping = {}
        
        for i, (node_id, node_data) in enumerate(G.nodes(data=True)):
            node_mapping[node_id] = i
            # 构造节点特征向量（可根据实际需求自定义）
            feature = np.zeros(self.config.model.hidden_dim)
            # 类型one-hot编码
            type_idx = ['action', 'object', 'task', 'constraint', 'safety'].index(
                node_data.get('type', 'object')
            )
            feature[type_idx] = 1.0
            node_features.append(feature)
        
        node_features = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # 边索引和边特征
        edge_index = []
        edge_features = []
        
        for u, v, edge_data in G.edges(data=True):
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                # 构造边特征向量（可根据实际需求自定义）
                edge_feat = np.zeros(self.config.graph.edge_hidden_dim)
                # 这里可添加关系类型编码等
                edge_features.append(edge_feat)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        edge_features = torch.tensor(np.array(edge_features), dtype=torch.float32)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'node_mapping': node_mapping
        }
    
    def create_dataset(self, data_dir: str) -> FrankaKGDataset:
        """
        从目录创建知识图谱数据集对象
        参数:
            data_dir: 数据目录路径
        返回:
            FrankaKGDataset: 数据集对象
        """
        # 加载所有原始数据
        all_data = self.load_data_from_directory(data_dir)
        # 合并实体和关系
        entities = self.merge_entities(all_data)
        relations = self.merge_relations(all_data)
        # 构建知识图谱
        graph = self.build_knowledge_graph(entities, relations)
        # 提取特征
        graph_features = self.extract_graph_features(graph)
        # 构造数据集
        dataset = FrankaKGDataset(
            graph=graph,
            features=graph_features,
            entities=entities,
            relations=relations,
            config=self.config
        )
        return dataset
    
    def create_data_loaders(self, 
                          train_dir: str,
                          val_dir: Optional[str] = None,
                          test_dir: Optional[str] = None,
                          batch_size: Optional[int] = None) -> Dict[str, DataLoader]:
        """
        创建训练、验证、测试的数据加载器
        参数:
            train_dir: 训练集目录
            val_dir: 验证集目录（可选）
            test_dir: 测试集目录（可选）
            batch_size: 批大小（可选，默认取配置）
        返回:
            dict: {'train':..., 'val':..., 'test':...}
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        loaders = {}
        
        # 训练集加载器
        train_dataset = self.create_dataset(train_dir)
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # 验证集加载器
        if val_dir:
            val_dataset = self.create_dataset(val_dir)
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # 测试集加载器
        if test_dir:
            test_dataset = self.create_dataset(test_dir)
            loaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        return loaders
    
    def save_processed_data(self, save_path: str):
        """
        保存处理好的实体、关系和图结构，便于后续复用
        参数:
            save_path: 保存文件路径
        """
        data_to_save = {
            'entities': self.entities,
            'relations': self.relations,
            'graph': nx.node_link_data(self.graph)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    
    def load_processed_data(self, load_path: str):
        """
        加载之前保存的实体、关系和图结构
        参数:
            load_path: 加载文件路径
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.entities = data['entities']
        self.relations = data['relations']
        self.graph = nx.node_link_graph(data['graph'])

    def visualize_knowledge_graph(self, save_path: Optional[str] = None):
        """
        可视化构建的知识图谱，并保存为HTML文件
        参数:
            save_path: 可选，保存路径，默认为kg_visualization.html
        返回:
            net: 可视化对象
        """
        from utils.kg_visualization import KnowledgeGraphVisualizer

        visualizer = KnowledgeGraphVisualizer(self.config)

        # 创建可视化
        net = visualizer.create_from_model_output(
            graph=self.graph,
            entities=self.entities,
            relations=self.relations
        )

        # 添加Franka特定组件（如机械臂结构等）
        visualizer.add_franka_specific_components(net)

        # 保存可视化结果
        if save_path:
            visualizer.save_with_legend(net, save_path)
        else:
            visualizer.save_with_legend(net, "kg_visualization.html")

        return net