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
    """Knowledge Graph Data Loader for MoE-KGC"""
    
    def __init__(self, config):
        self.config = config
        
        # 初始化处理器 - 这是关键！
        self.pdf_processor = PDFProcessor()
        self.csv_processor = CSVProcessor()
        self.yaml_processor = YAMLProcessor()
        
        # 初始化其他属性
        self.graph = nx.DiGraph()
        self.entities = {}
        self.relations = []
        
        print("KGDataLoader初始化完成")
        print(f"  PDF处理器: {self.pdf_processor}")
        print(f"  CSV处理器: {self.csv_processor}")
        print(f"  YAML处理器: {self.yaml_processor}")
    
    def load_data_from_directory(self, data_dir: str) -> Dict[str, Any]:
        """Load all data from directory"""
        data_dir = Path(data_dir)
        all_data = {
            'pdf_data': [],
            'csv_data': [],
            'yaml_data': []
        }
        
        print(f"\n开始加载数据目录: {data_dir}")
        
        # Process PDF files
        pdf_dir = data_dir / 'pdfs'
        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))
            print(f"找到 {len(pdf_files)} 个PDF文件")
            for pdf_file in pdf_files:
                try:
                    print(f"  处理: {pdf_file.name}")
                    pdf_data = self.pdf_processor.process(str(pdf_file))
                    if pdf_data:
                        all_data['pdf_data'].append(pdf_data)
                        print(f"    ✓ 成功")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")
        
        # Process CSV files  
        csv_dir = data_dir / 'csvs'
        if csv_dir.exists():
            csv_files = list(csv_dir.glob("*.csv"))
            print(f"\n找到 {len(csv_files)} 个CSV文件")
            # 只处理前几个避免太多输出
            for i, csv_file in enumerate(csv_files[:5]):
                try:
                    print(f"  处理: {csv_file.name}")
                    csv_data = self.csv_processor.process(str(csv_file))
                    if csv_data:
                        all_data['csv_data'].append(csv_data)
                        # 打印实体数量
                        entities = csv_data.get('entities', {})
                        if isinstance(entities, dict):
                            total = sum(len(v) for v in entities.values())
                        else:
                            total = len(entities)
                        print(f"    ✓ 成功 - {total} 个实体")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")
            
            # 处理剩余的CSV文件（不打印详情）
            for csv_file in csv_files[5:]:
                try:
                    csv_data = self.csv_processor.process(str(csv_file))
                    if csv_data:
                        all_data['csv_data'].append(csv_data)
                except:
                    pass
        
        # Process YAML files
        yaml_dir = data_dir / 'yamls'
        if yaml_dir.exists():
            yaml_files = list(yaml_dir.glob("*.yaml")) + list(yaml_dir.glob("*.yml"))
            print(f"\n找到 {len(yaml_files)} 个YAML文件")
            for yaml_file in yaml_files:
                try:
                    print(f"  处理: {yaml_file.name}")
                    yaml_data = self.yaml_processor.process(str(yaml_file))
                    if yaml_data:
                        all_data['yaml_data'].append(yaml_data)
                        entities = yaml_data.get('entities', [])
                        relations = yaml_data.get('relations', [])
                        print(f"    ✓ 成功 - {len(entities)} 个实体, {len(relations)} 个关系")
                except Exception as e:
                    print(f"    ✗ 失败: {e}")
        
        print(f"\n数据加载完成:")
        print(f"  PDF: {len(all_data['pdf_data'])} 个文件")
        print(f"  CSV: {len(all_data['csv_data'])} 个文件")
        print(f"  YAML: {len(all_data['yaml_data'])} 个文件")
        
        return all_data
    
    def merge_entities(self, all_data: Dict[str, List]) -> Dict[str, List]:
        """Merge entities from different sources - 更健壮的版本"""
        merged_entities = {}
    
        # 处理不同数据源
        for data_type in ['csv_data', 'yaml_data', 'pdf_data']:
            for data_item in all_data.get(data_type, []):
                entities = data_item.get('entities', [])
            
                # 处理统一的列表格式（新格式）
                if isinstance(entities, list):
                    for entity in entities:
                        # 确保是字典格式
                        if isinstance(entity, dict):
                            entity_type = entity.get('type', 'unknown')
                            if entity_type not in merged_entities:
                                merged_entities[entity_type] = []
                            merged_entities[entity_type].append(entity)
                        elif isinstance(entity, tuple) and len(entity) >= 3:
                            # 兼容旧的tuple格式
                            entity_dict = {
                                'id': f"legacy_entity_{len(merged_entities)}",
                                'text': entity[0],
                                'start': entity[1],
                                'end': entity[2],
                                'type': 'unknown'
                            }
                            if 'unknown' not in merged_entities:
                                merged_entities['unknown'] = []
                            merged_entities['unknown'].append(entity_dict)
            
                # 处理按类型分组的字典格式（旧格式）
                elif isinstance(entities, dict):
                    for entity_type, entity_list in entities.items():
                        # 映射实体类型
                        mapped_type = {
                            'tasks': 'task',
                            'actions': 'action',
                            'objects': 'object',
                            'constraints': 'constraint',
                            'safety': 'safety'
                        }.get(entity_type, entity_type)
                    
                        if mapped_type not in merged_entities:
                            merged_entities[mapped_type] = []
                    
                        if isinstance(entity_list, list):
                            for item in entity_list:
                                # 处理字典格式
                                if isinstance(item, dict):
                                    merged_entities[mapped_type].append(item)
                                # 处理tuple格式（PDF的旧格式）
                                elif isinstance(item, tuple) and len(item) >= 3:
                                    entity_dict = {
                                        'id': f"{mapped_type}_{item[0]}_{len(merged_entities[mapped_type])}",
                                        'type': mapped_type,
                                        'text': item[0],
                                        'name': item[0],
                                        'attributes': {
                                            'start_pos': item[1],
                                            'end_pos': item[2]
                                        }
                                    }
                                    merged_entities[mapped_type].append(entity_dict)
    
        # 去重
        for entity_type in merged_entities:
            seen = set()
            unique = []
            for entity in merged_entities[entity_type]:
                if isinstance(entity, dict):
                    entity_id = entity.get('id', str(entity.get('name', str(entity))))
                    if entity_id not in seen:
                        seen.add(entity_id)
                        unique.append(entity)
            merged_entities[entity_type] = unique
    
        print(f"\n合并后的实体统计:")
        for entity_type, entities in merged_entities.items():
            print(f"  {entity_type}: {len(entities)} 个")
    
        return merged_entities
    
    def merge_relations(self, all_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Merge relations from different sources"""
        all_relations = []
    
        # 处理不同数据源
        for data_type in ['csv_data', 'yaml_data', 'pdf_data']:
            for data_item in all_data.get(data_type, []):
                relations = data_item.get('relations', [])
            
                # 如果relations是列表（统一格式）
                if isinstance(relations, list):
                    all_relations.extend(relations)
            
                # 如果relations是字典（按类型分组）
                elif isinstance(relations, dict):
                    for rel_type, rel_list in relations.items():
                        if isinstance(rel_list, list):
                            # 为每个关系添加类型信息（如果还没有）
                            for rel in rel_list:
                                if isinstance(rel, dict) and 'type' not in rel:
                                    rel['type'] = rel_type
                            all_relations.extend(rel_list)
    
        # 去重（基于head、tail和type）
        seen = set()
        unique_relations = []
        for rel in all_relations:
            if isinstance(rel, dict):
                # 创建唯一标识
                rel_key = (
                    rel.get('head', ''), 
                    rel.get('tail', ''),
                    rel.get('type', '')
                )
                if rel_key not in seen:
                    seen.add(rel_key)
                    unique_relations.append(rel)
    
        print(f"\n合并后的关系数量: {len(unique_relations)}")
    
        # 统计关系类型
        rel_types = {}
        for rel in unique_relations:
            rel_type = rel.get('type', 'unknown')
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
    
        print("关系类型分布:")
        for rel_type, count in rel_types.items():
            print(f"  {rel_type}: {count} 个")
    
        return unique_relations
    
    
    
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
            feature = np.zeros(self.config.hidden_dim)
            # 类型one-hot编码
            type_idx = ['action', 'object', 'task', 'constraint', 'safety', 'spatial'].index(
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
        
        num_nodes = node_features.shape[0]
        seq_len = 128
        tabular_dim = 3
        task_dim = 256
        constraint_dim = 256
        safety_dim = 256

        # 文本模态（BERT输入格式）
        text_inputs = {
            'input_ids': torch.zeros(num_nodes, seq_len, dtype=torch.long),
            'attention_mask': torch.ones(num_nodes, seq_len, dtype=torch.long)
        }
        # 表格模态
        tabular_inputs = {
            'numerical': torch.zeros(num_nodes, tabular_dim),
            'categorical': {
                'action': torch.zeros(num_nodes, dtype=torch.long),
                'object_id': torch.zeros(num_nodes, dtype=torch.long)
            }
        }
        # 结构化模态
        structured_inputs = {
            'task_features': torch.zeros(num_nodes, task_dim),
            'constraint_features': torch.zeros(num_nodes, constraint_dim),
            'safety_features': torch.zeros(num_nodes, safety_dim)
        }
    
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'node_mapping': node_mapping,
            'text_inputs': text_inputs,
            'tabular_inputs': tabular_inputs,
            'structured_inputs': structured_inputs
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