#!/usr/bin/env python3
"""Evaluation script for MoE-KGC model"""

import argparse
import torch
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from data import KGDataLoader
from models import MoEKGC
from evaluation import Evaluator, BaselineComparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate MoE-KGC model')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--task', type=str, default='link_prediction',
                        choices=['link_prediction', 'entity_classification', 'relation_extraction'],
                        help='Task to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--compare_baselines', action='store_true',
                        help='Compare with baseline models')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save model predictions')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load configuration
    config = get_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print("Loading model...")
    model = MoEKGC(config)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load test data
    print("Loading test data...")
    data_loader = KGDataLoader(config)
    test_dataset = data_loader.create_dataset(args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Initialize evaluator
    evaluator = Evaluator(model, config, device)

    # Evaluate model
    print(f"Evaluating on {args.task} task...")
    results = evaluator.evaluate(
        test_loader,
        task=args.task,
        save_predictions=args.save_predictions
    )

    # Print results
    print("\n" + results['report'])

    # Save results
    results_path = output_dir / f'{args.task}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save report
    report_path = output_dir / f'{args.task}_report.txt'
    with open(report_path, 'w') as f:
        f.write(results['report'])

    # Visualize results
    viz_path = output_dir / f'{args.task}_visualization.png'
    evaluator.visualize_results(results['metrics'], save_path=str(viz_path))

    # Compare with baselines if requested
    if args.compare_baselines:
        print("\nComparing with baseline models...")
        comparison = BaselineComparison(model, config, device)

        comparison_df = comparison.compare_models(test_loader, task=args.task)

        # Save comparison results
        comparison_path = output_dir / 'baseline_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison results saved to {comparison_path}")

        # Generate comparison report
        comparison_report = comparison.generate_comparison_report(comparison_df)
        print("\n" + comparison_report)

        report_path = output_dir / 'comparison_report.txt'
        with open(report_path, 'w') as f:
            f.write(comparison_report)

        # Visualize comparison
        viz_path = output_dir / 'comparison_visualization.png'
        comparison.visualize_comparison(comparison_df, save_path=str(viz_path))


if __name__ == '__main__':
    main()