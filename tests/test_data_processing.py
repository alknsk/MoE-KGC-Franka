import unittest
import torch
import pandas as pd
import tempfile
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessors import PDFProcessor, CSVProcessor, YAMLProcessor
from data import KGDataLoader, FrankaKGDataset
from config import get_config


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = get_config()

    def test_csv_processor(self):
        """Test CSVProcessor"""
        processor = CSVProcessor()

        # Create dummy CSV data
        data = {
            'timestamp': [0.0, 0.1, 0.2, 0.3],
            'action': ['grasp', 'move', 'place', 'release'],
            'joint_positions': ['[1.0,2.0,3.0,4.0,5.0,6.0,7.0]'] * 4,
            'gripper_state': [0, 0, 1, 1],
            'object_id': ['obj1', 'obj1', 'obj1', 'obj1'],
            'success': [1, 1, 1, 0]
        }
        df = pd.DataFrame(data)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            df.to_csv(tmp.name, index=False)

            # Process CSV
            result = processor.process(tmp.name)

        # Check results
        self.assertIn('entities', result)
        self.assertIn('relations', result)
        self.assertIn('statistics', result)

        # Check entities
        self.assertEqual(len(result['entities']['actions']), 4)
        self.assertEqual(len(result['entities']['objects']), 1)

        # Check relations
        self.assertEqual(len(result['relations']['temporal']), 3)  # n-1 temporal relations
        self.assertEqual(len(result['relations']['interactions']), 4)  # all actions interact with object

        # Check statistics
        self.assertEqual(result['statistics']['num_actions'], 4)
        self.assertEqual(result['statistics']['success_rate'], 0.75)

    def test_yaml_processor(self):
        """Test YAMLProcessor"""
        processor = YAMLProcessor()

        # Create dummy YAML data
        data = {
            'tasks': [
                {
                    'name': 'pick_and_place',
                    'type': 'manipulation',
                    'parameters': {'object': 'cube', 'target': 'box'},
                    'constraints': [{'type': 'collision_free'}],
                    'safety_limits': {'max_force': 50.0}
                }
            ],
            'global_constraints': [
                {'name': 'workspace_limit', 'type': 'spatial'}
            ],
            'safety_limits': {
                'max_joint_velocity': 1.0,
                'max_force': 100.0
            }
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(data, tmp)

            # Process YAML
            result = processor.process(tmp.name)

        # Check results
        self.assertIn('entities', result)
        self.assertEqual(len(result['entities']['tasks']), 1)
        self.assertEqual(len(result['entities']['constraints']), 2)  # 1 task + 1 global
        self.assertEqual(len(result['entities']['safety']), 2)

    def test_kg_dataset(self):
        """Test FrankaKGDataset"""
        import networkx as nx

        # Create dummy graph
        G = nx.DiGraph()
        G.add_nodes_from([
            (0, {'type': 'action', 'name': 'grasp'}),
            (1, {'type': 'object', 'name': 'cube'}),
            (2, {'type': 'action', 'name': 'move'})
        ])
        G.add_edges_from([
            (0, 1, {'type': 'interacts_with'}),
            (0, 2, {'type': 'follows'})
        ])

        # Create features
        features = {
            'node_features': torch.randn(3, 256),
            'edge_index': torch.tensor([[0, 0], [1, 2]]),
            'edge_features': torch.randn(2, 128),
            'node_mapping': {0: 0, 1: 1, 2: 2},
            'text_inputs': {
                'input_ids': torch.zeros(3, 32, dtype=torch.long),
                'attention_mask': torch.ones(3, 32, dtype=torch.long)
            },
            'tabular_inputs': {
                'numerical': torch.zeros(3, 3),
                'categorical': {
                    'action': torch.zeros(3, dtype=torch.long),
                    'object_id': torch.zeros(3, dtype=torch.long)
                }
            },
            'structured_inputs': {
                'task_features': torch.zeros(3, 256),
                'constraint_features': torch.zeros(3, 256),
                'safety_features': torch.zeros(3, 256)
            }
        }

        # Create dataset
        dataset = FrankaKGDataset(
            graph=G,
            features=features,
            entities={'action': [0, 2], 'object': [1]},
            relations=[
                {'head': 0, 'tail': 1, 'type': 'interacts_with'},
                {'head': 0, 'tail': 2, 'type': 'follows'}
            ],
            config=self.config,
            task='link_prediction'
        )

        # Test dataset
        self.assertGreater(len(dataset), 0)

        # Test get item
        sample = dataset[0]
        self.assertIn('node_features', sample)
        self.assertIn('edge_index', sample)

        # Test statistics
        stats = dataset.get_statistics()
        self.assertEqual(stats['num_nodes'], 3)
        self.assertEqual(stats['num_edges'], 2)

    def test_data_loader(self):
        """Test KGDataLoader"""
        loader = KGDataLoader(self.config)

        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy CSV
            csv_data = pd.DataFrame({
                'timestamp': [0.0],
                'action': ['grasp'],
                'joint_positions': ['[0,0,0,0,0,0,0]'],
                'gripper_state': [0],
                'object_id': ['obj1'],
                'success': [1]
            })
            csv_path = Path(tmpdir) / 'test.csv'
            csv_data.to_csv(csv_path, index=False)

            # Create dummy YAML
            yaml_data = {
                'tasks': [{'name': 'test', 'type': 'test', 'parameters': {},
                           'constraints': [], 'safety_limits': {}}]
            }
            yaml_path = Path(tmpdir) / 'test.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f)

            # Load data
            all_data = loader.load_data_from_directory(tmpdir)

            # Check loaded data
            self.assertEqual(len(all_data['csv_data']), 1)
            self.assertEqual(len(all_data['yaml_data']), 1)

            # Test entity merging
            entities = loader.merge_entities(all_data)
            self.assertIn('action', entities)
            self.assertIn('task', entities)


if __name__ == '__main__':
    unittest.main()