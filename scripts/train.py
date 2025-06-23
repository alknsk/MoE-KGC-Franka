#!/usr/bin/env python3
"""Training script for MoE-KGC model"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data import KGDataLoader
from models import MoEKGC
from training import Trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Train MoE-KGC model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--task', type=str, default='link_prediction',
                        choices=['link_prediction', 'entity_classification', 'relation_extraction'],
                        help='Task to train on')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Load configuration
    config = get_config(args.config)

    # Override config with command line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading data...")
    data_loader = KGDataLoader(config)

    # Prepare data directories
    train_dir = Path(args.data_dir) / 'train'
    val_dir = Path(args.data_dir) / 'val'

    if not train_dir.exists():
        raise ValueError(f"Training data directory not found: {train_dir}")

    # Create data loaders
    loaders = data_loader.create_data_loaders(
        train_dir=str(train_dir),
        val_dir=str(val_dir) if val_dir.exists() else None,
        batch_size=config.training.batch_size
    )

    # Initialize model
    print("Initializing model...")
    model = MoEKGC(config)

    # Initialize trainer
    trainer = Trainer(model, config, device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders.get('val'),
        task=args.task,
        save_dir=args.save_dir
    )

    # Print final results
    print("\nTraining completed!")
    print(f"Best validation score: {trainer.best_val_score:.4f}")

    # Save training history
    import json
    history_path = Path(args.save_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    main()