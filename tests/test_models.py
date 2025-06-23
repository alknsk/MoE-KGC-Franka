import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models import MoEKGC
from models.experts import ActionExpert, SpatialExpert, TemporalExpert, SemanticExpert, SafetyExpert
from models.gating import AdaptiveGating
from models.graph_layers import EnhancedGNN, GraphFusion


class TestModels(unittest.TestCase):
    """Test cases for model components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = get_config()
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.seq_len = 10
        self.hidden_dim = 256
        self.num_nodes = 20

    def test_action_expert(self):
        """Test ActionExpert"""
        expert = ActionExpert(
            input_dim=self.hidden_dim,
            hidden_dims=[128, 64],
            output_dim=self.hidden_dim
        )

        # Test forward pass
        x = torch.randn(self.batch_size, self.hidden_dim)
        output = expert(x)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_spatial_expert(self):
        """Test SpatialExpert"""
        expert = SpatialExpert(
            input_dim=self.hidden_dim,
            hidden_dims=[128, 64],
            output_dim=self.hidden_dim
        )

        # Test with position data
        x = torch.randn(self.batch_size, self.hidden_dim)
        positions = torch.randn(self.batch_size, 3)
        output = expert(x, positions=positions)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())

        # Test spatial relation computation
        obj1 = torch.randn(self.batch_size, self.hidden_dim)
        obj2 = torch.randn(self.batch_size, self.hidden_dim)
        relation_output = expert.compute_spatial_relation(obj1, obj2)

        self.assertIn('probabilities', relation_output)
        self.assertEqual(relation_output['probabilities'].shape[0], self.batch_size)

    def test_temporal_expert(self):
        """Test TemporalExpert"""
        expert = TemporalExpert(
            input_dim=self.hidden_dim,
            hidden_dims=[128, 64],
            output_dim=self.hidden_dim
        )

        # Test with sequence data
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        timestamps = torch.randn(self.batch_size, self.seq_len)
        output = expert(x, timestamps=timestamps)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_semantic_expert(self):
        """Test SemanticExpert"""
        expert = SemanticExpert(
            input_dim=self.hidden_dim,
            hidden_dims=[128, 64],
            output_dim=self.hidden_dim,
            vocab_size=1000
        )

        # Test with word indices
        x = torch.randn(self.batch_size, self.hidden_dim)
        word_indices = torch.randint(0, 1000, (self.batch_size, self.seq_len))
        output = expert(x, word_indices=word_indices)

        self.assertEqual(output.shape, (self.batch_size, self.hidden_dim))
        self.assertFalse(torch.isnan(output).any())

    def test_safety_expert(self):
        """Test SafetyExpert"""
        expert = SafetyExpert(
            input_dim=self.hidden_dim,
            hidden_dims=[128, 64],
            output_dim=self.hidden_dim
        )

        # Test collision risk assessment
        state = torch.randn(self.batch_size, self.hidden_dim)
        risk_output = expert.assess_collision_risk(state)

        self.assertIn('collision_probability', risk_output)
        self.assertEqual(risk_output['collision_probability'].shape, (self.batch_size, 1))

        # Test safety level classification
        safety_output = expert.classify_safety_level(state)
        self.assertIn('safety_level', safety_output)
        self.assertEqual(safety_output['safety_level'].shape, (self.batch_size,))

    def test_adaptive_gating(self):
        """Test AdaptiveGating"""
        gating = AdaptiveGating(
            input_dim=self.hidden_dim,
            num_experts=5,
            top_k=2
        )

        # Test forward pass
        x = torch.randn(self.batch_size, self.hidden_dim)
        output = gating(x)

        self.assertIn('gates', output)
        self.assertIn('indices', output)
        self.assertEqual(output['gates'].shape, (self.batch_size, 2))
        self.assertEqual(output['indices'].shape, (self.batch_size, 2))

        # Check gate values sum to 1
        gate_sums = output['gates'].sum(dim=-1)
        self.assertTrue(torch.allclose(gate_sums, torch.ones(self.batch_size), atol=1e-6))

    def test_enhanced_gnn(self):
        """Test EnhancedGNN"""
        gnn = EnhancedGNN(
            input_dim=self.hidden_dim,
            hidden_dim=128,
            output_dim=self.hidden_dim,
            num_layers=2
        )

        # Create graph data
        x = torch.randn(self.num_nodes, self.hidden_dim)
        edge_index = torch.randint(0, self.num_nodes, (2, 30))

        # Test forward pass
        output = gnn(x, edge_index)

        self.assertIn('node_embeddings', output)
        self.assertIn('graph_embedding', output)
        self.assertEqual(output['node_embeddings'].shape, (self.num_nodes, self.hidden_dim))
        self.assertEqual(output['graph_embedding'].shape, (1, self.hidden_dim))

    def test_graph_fusion(self):
        """Test GraphFusion"""
        fusion = GraphFusion(
            input_dims=[self.hidden_dim] * 3,
            output_dim=self.hidden_dim,
            fusion_type='attention'
        )

        # Test fusion of multiple inputs
        inputs = [torch.randn(self.batch_size, self.hidden_dim) for _ in range(3)]
        output = fusion(inputs)

        self.assertIn('fused_output', output)
        self.assertEqual(output['fused_output'].shape, (self.batch_size, self.hidden_dim))

    def test_moe_kgc_model(self):
        """Test complete MoEKGC model"""
        model = MoEKGC(self.config)

        # Create dummy batch
        batch = {
            'text_inputs': {
                'input_ids': torch.randint(0, 1000, (self.batch_size, 128)),
                'attention_mask': torch.ones(self.batch_size, 128)
            },
            'node_features': torch.randn(self.num_nodes, self.hidden_dim),
            'edge_index': torch.randint(0, self.num_nodes, (2, 50)),
            'head_embeddings': torch.randn(self.batch_size, self.hidden_dim),
            'tail_embeddings': torch.randn(self.batch_size, self.hidden_dim),
            'relation_ids': torch.randint(0, 50, (self.batch_size,)),
            'labels': torch.randint(0, 2, (self.batch_size,)).float()
        }

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(batch, task='link_prediction')

        self.assertIn('scores', output)
        self.assertEqual(output['scores'].shape, (self.batch_size,))
        self.assertFalse(torch.isnan(output['scores']).any())

    def test_model_save_load(self):
        """Test model saving and loading"""
        model1 = MoEKGC(self.config)

        # Save model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            torch.save(model1.state_dict(), tmp.name)

            # Load model
            model2 = MoEKGC(self.config)
            model2.load_state_dict(torch.load(tmp.name))

            # Check parameters are the same
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.allclose(p1, p2))


if __name__ == '__main__':
    unittest.main()