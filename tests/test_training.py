import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from pathlib import Path
import sys
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config
from models import MoEKGC
from training import Trainer, MultiTaskLoss, FocalLoss, ContrastiveLoss
from training.metrics import Metrics, compute_metrics


class TestTraining(unittest.TestCase):
    """Test cases for training components"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = get_config()
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.num_samples = 20

        # Reduce model size for testing
        self.config.hidden_dim = 64
        self.config.expert_hidden_dim = 32
        self.config.training.epochs = 2

    def create_dummy_dataloader(self, task='link_prediction'):
        """Create dummy dataloader for testing"""
        if task == 'link_prediction':
            # Create dummy link prediction data
            head_embeddings = torch.randn(self.num_samples, self.config.hidden_dim)
            tail_embeddings = torch.randn(self.num_samples, self.config.hidden_dim)
            relation_ids = torch.randint(0, 10, (self.num_samples,))
            labels = torch.randint(0, 2, (self.num_samples,)).float()

            dataset = TensorDataset(head_embeddings, tail_embeddings, relation_ids, labels)

            def collate_fn(batch):
                return {
                    'head_embeddings': torch.stack([b[0] for b in batch]),
                    'tail_embeddings': torch.stack([b[1] for b in batch]),
                    'relation_ids': torch.stack([b[2] for b in batch]),
                    'labels': torch.stack([b[3] for b in batch]),
                    'node_features': torch.randn(10, self.config.hidden_dim),
                    'edge_index': torch.randint(0, 10, (2, 20))
                }

            return DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

        elif task == 'entity_classification':
            # Create dummy classification data
            node_features = torch.randn(self.num_samples, self.config.hidden_dim)
            labels = torch.randint(0, 5, (self.num_samples,))

            dataset = TensorDataset(node_features, labels)

            def collate_fn(batch):
                return {
                    'node_features': torch.stack([b[0] for b in batch]),
                    'labels': torch.stack([b[1] for b in batch]),
                    'edge_index': torch.randint(0, self.batch_size, (2, 10))
                }

            return DataLoader(dataset, batch_size=self.batch_size, collate_fn=collate_fn)

    def test_multi_task_loss(self):
        """Test MultiTaskLoss"""
        loss_fn = MultiTaskLoss(self.config)

        # Test link prediction loss
        outputs = {
            'scores': torch.randn(self.batch_size),
            'gating_loss': torch.tensor(0.01)
        }
        targets = {
            'labels': torch.randint(0, 2, (self.batch_size,)).float()
        }

        losses = loss_fn(outputs, targets, task='link_prediction')

        self.assertIn('total_loss', losses)
        self.assertIn('link_prediction_loss', losses)
        self.assertIn('gating_loss', losses)
        self.assertGreater(losses['total_loss'].item(), 0)

    def test_focal_loss(self):
        """Test FocalLoss"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

        inputs = torch.randn(self.batch_size, 5)
        targets = torch.randint(0, 5, (self.batch_size,))

        loss = focal_loss(inputs, targets)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)

    def test_contrastive_loss(self):
        """Test ContrastiveLoss"""
        contrastive_loss = ContrastiveLoss(temperature=0.07)

        anchor = torch.randn(self.batch_size, 64)
        positive = torch.randn(self.batch_size, 64)
        negative = torch.randn(self.batch_size, 64)

        # Test with explicit negative
        loss1 = contrastive_loss(anchor, positive, negative)
        self.assertIsInstance(loss1, torch.Tensor)
        self.assertGreater(loss1.item(), 0)

        # Test without explicit negative (SimCLR style)
        loss2 = contrastive_loss(anchor, positive)
        self.assertIsInstance(loss2, torch.Tensor)
        self.assertGreater(loss2.item(), 0)

    def test_metrics_computation(self):
        """Test metrics computation"""
        metrics = Metrics(self.config)

        # Test link prediction metrics
        scores = torch.tensor([0.8, -0.5, 0.3, -0.2])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

        link_metrics = metrics.compute_link_prediction_metrics(scores, labels)

        self.assertIn('accuracy', link_metrics)
        self.assertIn('precision', link_metrics)
        self.assertIn('recall', link_metrics)
        self.assertIn('f1', link_metrics)

        # Test classification metrics
        logits = torch.randn(10, 5)
        labels = torch.randint(0, 5, (10,))

        class_metrics = metrics.compute_classification_metrics(logits, labels)

        self.assertIn('accuracy', class_metrics)
        self.assertIn('macro_f1', class_metrics)
        self.assertIn('weighted_f1', class_metrics)

    def test_trainer_initialization(self):
        """Test Trainer initialization"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        self.assertIsInstance(trainer.model, nn.Module)
        self.assertIsInstance(trainer.optimizer, torch.optim.Optimizer)
        self.assertIsInstance(trainer.criterion, nn.Module)
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.global_step, 0)

    def test_train_epoch(self):
        """Test single epoch training"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        train_loader = self.create_dummy_dataloader('link_prediction')

        # Train one epoch
        epoch_results = trainer.train_epoch(train_loader, task='link_prediction')

        self.assertIn('loss', epoch_results)
        self.assertIsInstance(epoch_results['loss'], float)
        self.assertGreater(epoch_results['loss'], 0)

        # Check global step increased
        self.assertGreater(trainer.global_step, 0)

    def test_validation(self):
        """Test validation"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        val_loader = self.create_dummy_dataloader('link_prediction')

        # Validate
        val_results = trainer.validate(val_loader, task='link_prediction')

        self.assertIn('loss', val_results)
        self.assertIn('accuracy', val_results)
        self.assertIsInstance(val_results['loss'], float)

    def test_full_training(self):
        """Test full training loop"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        train_loader = self.create_dummy_dataloader('link_prediction')
        val_loader = self.create_dummy_dataloader('link_prediction')

        with tempfile.TemporaryDirectory() as tmpdir:
            # Train for 2 epochs
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                task='link_prediction',
                save_dir=tmpdir
            )

            # Check history
            self.assertIn('train_loss', history)
            self.assertIn('val_loss', history)
            self.assertEqual(len(history['train_loss']), 2)
            self.assertEqual(len(history['val_loss']), 2)

            # Check model saved
            checkpoint_path = Path(tmpdir) / 'final_model.pt'
            self.assertTrue(checkpoint_path.exists())

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        # Modify trainer state
        trainer.epoch = 5
        trainer.global_step = 100
        trainer.best_val_score = 0.95

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            # Save checkpoint
            trainer.save_checkpoint(tmp.name)

            # Create new trainer and load checkpoint
            model2 = MoEKGC(self.config)
            trainer2 = Trainer(model2, self.config, self.device)
            trainer2.load_checkpoint(tmp.name)

            # Check state restored
            self.assertEqual(trainer2.epoch, 5)
            self.assertEqual(trainer2.global_step, 100)
            self.assertEqual(trainer2.best_val_score, 0.95)

    def test_gradient_clipping(self):
        """Test gradient clipping"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        # Set gradient clipping
        self.config.training.gradient_clip = 1.0

        # Create large gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 10

        # Get gradient norms before clipping
        total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

        # Reset gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 10

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clip)

        # Get gradient norms after clipping
        total_norm_after = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm_after += param.grad.data.norm(2).item() ** 2
        total_norm_after = total_norm_after ** 0.5

        # Check clipping worked
        self.assertLessEqual(total_norm_after, self.config.training.gradient_clip + 1e-6)

    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling"""
        model = MoEKGC(self.config)
        trainer = Trainer(model, self.config, self.device)

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Step scheduler
        trainer.scheduler.step()

        new_lr = trainer.optimizer.param_groups[0]['lr']

        # For cosine annealing, LR should decrease
        if self.config.training.scheduler == 'cosine':
            self.assertLess(new_lr, initial_lr)


if __name__ == '__main__':
    unittest.main()