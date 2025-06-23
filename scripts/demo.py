#!/usr/bin/env python3
"""Demo script for MoE-KGC model"""

import argparse
import torch
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from models import MoEKGC
from data.preprocessors import PDFProcessor, CSVProcessor, YAMLProcessor


class MoEKGCDemo:
    """Interactive demo for MoE-KGC model"""

    def __init__(self, model_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load configuration
        self.config = get_config(config_path)

        # Initialize model
        self.model = MoEKGC(self.config)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Initialize processors
        self.pdf_processor = PDFProcessor()
        self.csv_processor = CSVProcessor()
        self.yaml_processor = YAMLProcessor()

        print(f"Model loaded successfully on {self.device}")

    def process_pdf(self, pdf_path: str):
        """Demo PDF processing"""
        print(f"\nProcessing PDF: {pdf_path}")

        # Process PDF
        pdf_data = self.pdf_processor.process(pdf_path)

        # Extract entities and relations
        print(f"Extracted {len(pdf_data['entities'])} entity types")
        print(f"Extracted {len(pdf_data['relations'])} relations")

        # Show sample entities
        print("\nSample entities:")
        for entity_type, entities in pdf_data['entities'].items():
            if entities:
                print(f"  {entity_type}: {entities[0][0] if isinstance(entities[0], tuple) else entities[0]}")

        return pdf_data

    def process_csv(self, csv_path: str):
        """Demo CSV processing"""
        print(f"\nProcessing CSV: {csv_path}")

        # Process CSV
        csv_data = self.csv_processor.process(csv_path)

        # Show statistics
        stats = csv_data['statistics']
        print(f"\nStatistics:")
        print(f"  Number of actions: {stats['num_actions']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Unique actions: {stats['num_unique_actions']}")

        return csv_data

    def process_yaml(self, yaml_path: str):
        """Demo YAML processing"""
        print(f"\nProcessing YAML: {yaml_path}")

        # Process YAML
        yaml_data = self.yaml_processor.process(yaml_path)

        # Show extracted information
        print(f"\nExtracted:")
        print(f"  Tasks: {len(yaml_data['entities']['tasks'])}")
        print(f"  Constraints: {len(yaml_data['entities']['constraints'])}")
        print(f"  Safety limits: {len(yaml_data['entities']['safety'])}")

        return yaml_data

    def build_knowledge_graph(self, entities: dict, relations: list):
        """Build and visualize knowledge graph"""
        G = nx.DiGraph()

        # Add nodes
        node_id = 0
        node_mapping = {}
        for entity_type, entity_list in entities.items():
            for entity in entity_list[:10]:  # Limit for visualization
                node_name = entity.get('name', entity.get('text', f'node_{node_id}'))
                G.add_node(node_id, name=node_name, type=entity_type)
                node_mapping[node_name] = node_id
                node_id += 1

        # Add edges
        for relation in relations[:20]:  # Limit for visualization
            if 'head' in relation and 'tail' in relation:
                head = relation['head']
                tail = relation['tail']
                if isinstance(head, dict):
                    head = head.get('text', str(head))
                if isinstance(tail, dict):
                    tail = tail.get('text', str(tail))

                if head in node_mapping and tail in node_mapping:
                    G.add_edge(
                        node_mapping[head],
                        node_mapping[tail],
                        type=relation.get('type', 'related')
                    )

        return G

    def visualize_graph_interactive(self, G: nx.DiGraph, save_path: str = None):
        """使用pyvis创建交互式知识图谱"""
        from utils.kg_visualization import KnowledgeGraphVisualizer

        visualizer = KnowledgeGraphVisualizer(self.config)

        # 获取专家激活情况
        expert_activation = self.get_expert_activation_for_nodes(G)

        # 创建增强的可视化
        net = visualizer.create_from_model_output(
            graph=G,
            entities=self.entities,
            relations=self.relations,
            predictions=expert_activation
        )

        # 添加Franka组件
        visualizer.add_franka_specific_components(net)

        # 保存
        output_path = save_path or "franka_kg_moe_demo.html"
        visualizer.save_with_legend(net, output_path)

        print(f"交互式知识图谱已生成：{output_path}")

    def predict_link(self, head_entity: str, tail_entity: str, relation: str = None):
        """Demo link prediction"""
        print(f"\nPredicting link between '{head_entity}' and '{tail_entity}'")

        # This is a simplified demo - in practice, you'd need proper entity embeddings
        # Here we create dummy embeddings for demonstration
        dummy_head = torch.randn(1, self.config.model.hidden_dim).to(self.device)
        dummy_tail = torch.randn(1, self.config.model.hidden_dim).to(self.device)
        relation_id = torch.tensor([0]).to(self.device)  # Dummy relation ID

        with torch.no_grad():
            outputs = self.model.link_prediction_head(dummy_head, dummy_tail, relation_id)
            probability = outputs['probabilities'].item()

        print(f"Link probability: {probability:.3f}")
        print(f"Prediction: {'Likely' if probability > 0.5 else 'Unlikely'} to be connected")

        return probability

    def analyze_expert_activation(self, input_data):
        """Analyze which experts are activated for given input"""
        print("\nAnalyzing expert activation patterns...")

        # Create dummy input for demonstration
        dummy_input = torch.randn(1, self.config.model.expert_hidden_dim * 3).to(self.device)

        with torch.no_grad():
            gating_output = self.model.gating(dummy_input, return_all_scores=True)
            expert_scores = gating_output['all_scores'].cpu().numpy()[0]

        expert_names = ['Action', 'Spatial', 'Temporal', 'Semantic', 'Safety']

        # Plot expert activation
        plt.figure(figsize=(10, 6))
        plt.bar(expert_names, expert_scores)
        plt.title("Expert Activation Scores")
        plt.ylabel("Activation Score")
        plt.ylim(0, 1)

        for i, (name, score) in enumerate(zip(expert_names, expert_scores)):
            plt.text(i, score + 0.02, f'{score:.3f}', ha='center')

        plt.show()

        # Print top experts
        top_experts = gating_output['indices'].cpu().numpy()[0]
        print(f"\nTop activated experts: {[expert_names[i] for i in top_experts]}")


def main():
    parser = argparse.ArgumentParser(description='MoE-KGC Interactive Demo')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--pdf', type=str, help='PDF file to process')
    parser.add_argument('--csv', type=str, help='CSV file to process')
    parser.add_argument('--yaml', type=str, help='YAML file to process')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize knowledge graph')
    parser.add_argument('--predict', nargs=2, metavar=('HEAD', 'TAIL'),
                        help='Predict link between two entities')
    parser.add_argument('--analyze_experts', action='store_true',
                        help='Analyze expert activation patterns')

    args = parser.parse_args()

    # Initialize demo
    demo = MoEKGCDemo(args.model, args.config)

    # Process different file types
    all_entities = {}
    all_relations = []

    if args.pdf:
        pdf_data = demo.process_pdf(args.pdf)
        all_entities.update(pdf_data['entities'])
        all_relations.extend(pdf_data['relations'])

    if args.csv:
        csv_data = demo.process_csv(args.csv)
        all_entities.update(csv_data['entities'])
        all_relations.extend(csv_data['relations']['temporal'] +
                             csv_data['relations']['interactions'])

    if args.yaml:
        yaml_data = demo.process_yaml(args.yaml)
        all_entities.update(yaml_data['entities'])
        all_relations.extend(yaml_data['relations']['hierarchical'])

    # Build and visualize knowledge graph
    if args.visualize and (all_entities or all_relations):
        print("\nBuilding knowledge graph...")
        G = demo.build_knowledge_graph(all_entities, all_relations)
        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        demo.visualize_graph(G)

    # Predict link
    if args.predict:
        head, tail = args.predict
        demo.predict_link(head, tail)

    # Analyze expert activation
    if args.analyze_experts:
        demo.analyze_expert_activation(None)

    print("\nDemo completed!")


if __name__ == '__main__':
    main()