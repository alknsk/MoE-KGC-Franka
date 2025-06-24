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
    """Main data loader for knowledge graph construction"""
    
    def __init__(self, config):
        self.config = config
        self.pdf_processor = PDFProcessor()
        self.csv_processor = CSVProcessor()
        self.yaml_processor = YAMLProcessor()
        
        self.entities = {}
        self.relations = []
        self.graph = nx.MultiDiGraph()
        
    def load_data_from_directory(self, data_dir: str) -> Dict[str, Any]:
        """Load all data from directory"""
        data_dir = Path(data_dir)
        all_data = {
            'pdf_data': [],
            'csv_data': [],
            'yaml_data': []
        }
        
        # Process PDF files
        for pdf_file in data_dir.glob("*.pdf"):
            print(f"Processing PDF: {pdf_file}")
            pdf_data = self.pdf_processor.process(str(pdf_file))
            all_data['pdf_data'].append(pdf_data)
        
        # Process CSV files
        for csv_file in data_dir.glob("*.csv"):
            print(f"Processing CSV: {csv_file}")
            csv_data = self.csv_processor.process(str(csv_file))
            all_data['csv_data'].append(csv_data)
        
        # Process YAML files
        for yaml_file in data_dir.glob("*.yaml"):
            print(f"Processing YAML: {yaml_file}")
            yaml_data = self.yaml_processor.process(str(yaml_file))
            all_data['yaml_data'].append(yaml_data)
        
        return all_data
    
    def merge_entities(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Merge entities from different sources"""
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
        
        # From PDF data
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
        
        # From CSV data
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
        
        # From YAML data
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
        """Merge relations from different sources"""
        merged_relations = []
        
        # From PDF data
        for pdf_data in all_data['pdf_data']:
            for relation in pdf_data.get('relations', []):
                merged_relations.append({
                    'head': relation['head']['text'],
                    'tail': relation['tail']['text'],
                    'type': 'contextual',
                    'context': relation.get('context', ''),
                    'source': 'pdf'
                })
        
        # From CSV data
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
        
        # From YAML data
        for yaml_data in all_data['yaml_data']:
            relations = yaml_data.get('relations', {})
            
            for relation in relations.get('hierarchical', []):
                merged_relations.append({
                    **relation,
                    'source': 'yaml'
                })
        
        return merged_relations
    
    def build_knowledge_graph(self, entities: Dict[str, List], relations: List[Dict]) -> nx.MultiDiGraph:
        """Build NetworkX graph from entities and relations"""
        G = nx.MultiDiGraph()
        
        # Add nodes
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                G.add_node(
                    entity['id'],
                    **entity
                )
        
        # Add edges
        for relation in relations:
            G.add_edge(
                relation['head'],
                relation['tail'],
                **relation
            )
        
        return G
    
    def extract_graph_features(self, G: nx.MultiDiGraph) -> Dict[str, torch.Tensor]:
        """Extract features from graph for model input"""
        # Node features
        node_features = []
        node_mapping = {}
        
        for i, (node_id, node_data) in enumerate(G.nodes(data=True)):
            node_mapping[node_id] = i
            
            # Create feature vector (customize based on your needs)
            feature = np.zeros(self.config.model.hidden_dim)
            # Add type encoding
            type_idx = ['action', 'object', 'task', 'constraint', 'safety'].index(
                node_data.get('type', 'object')
            )
            feature[type_idx] = 1.0
            
            node_features.append(feature)
        
        node_features = torch.tensor(np.array(node_features), dtype=torch.float32)
        
        # Edge indices and features
        edge_index = []
        edge_features = []
        
        for u, v, edge_data in G.edges(data=True):
            if u in node_mapping and v in node_mapping:
                edge_index.append([node_mapping[u], node_mapping[v]])
                
                # Create edge feature vector
                edge_feat = np.zeros(self.config.graph.edge_hidden_dim)
                # Add relation type encoding
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
        """Create dataset from directory"""
        # Load all data
        all_data = self.load_data_from_directory(data_dir)
        
        # Merge entities and relations
        entities = self.merge_entities(all_data)
        relations = self.merge_relations(all_data)
        
        # Build knowledge graph
        graph = self.build_knowledge_graph(entities, relations)
        
        # Extract features
        graph_features = self.extract_graph_features(graph)
        
        # Create dataset
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
        """Create data loaders for training, validation, and testing"""
        if batch_size is None:
            batch_size = self.config.training.batch_size
        
        loaders = {}
        
        # Training data loader
        train_dataset = self.create_dataset(train_dir)
        loaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Validation data loader
        if val_dir:
            val_dataset = self.create_dataset(val_dir)
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Test data loader
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
        """Save processed data for later use"""
        data_to_save = {
            'entities': self.entities,
            'relations': self.relations,
            'graph': nx.node_link_data(self.graph)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
    
    def load_processed_data(self, load_path: str):
        """Load previously processed data"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.entities = data['entities']
        self.relations = data['relations']
        self.graph = nx.node_link_graph(data['graph'])

    def visualize_knowledge_graph(self, save_path: Optional[str] = None):
        """可视化构建的知识图谱"""
        from utils.kg_visualization import KnowledgeGraphVisualizer

        visualizer = KnowledgeGraphVisualizer(self.config)

        # 创建可视化
        net = visualizer.create_from_model_output(
            graph=self.graph,
            entities=self.entities,
            relations=self.relations
        )

        # 添加Franka特定组件
        visualizer.add_franka_specific_components(net)

        # 保存
        if save_path:
            visualizer.save_with_legend(net, save_path)
        else:
            visualizer.save_with_legend(net, "kg_visualization.html")

        return net