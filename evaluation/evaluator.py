import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from training.metrics import compute_metrics

class Evaluator:
    """Comprehensive evaluator for MoE-KGC model"""
    
    def __init__(self, model: nn.Module, config, device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader, task: str = 'link_prediction',
                save_predictions: bool = True) -> Dict[str, Any]:
        """Comprehensive evaluation on test set"""
        all_outputs = []
        all_targets = []
        all_predictions = []
        expert_utilization = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch, task=task)
                
                # Store outputs and targets
                all_outputs.append(outputs)
                all_targets.append(batch)
                
                # Store predictions
                if task == 'link_prediction':
                    predictions = {
                        'scores': outputs['scores'].cpu(),
                        'probabilities': outputs['probabilities'].cpu()
                    }
                elif task in ['entity_classification', 'relation_extraction']:
                    predictions = {
                        'predictions': outputs['predicted_class'].cpu() 
                                     if 'predicted_class' in outputs 
                                     else outputs['predicted_relation'].cpu(),
                        'probabilities': outputs['probabilities'].cpu()
                    }
                all_predictions.append(predictions)
                
                # Track expert utilization
                if 'expert_utilization' in outputs and outputs['expert_utilization'] is not None:
                    expert_utilization.append(outputs['expert_utilization'].cpu())
        
        # Aggregate metrics
        metrics = self._compute_aggregate_metrics(all_outputs, all_targets, task)
        
        # Analyze expert utilization
        if expert_utilization:
            expert_analysis = self._analyze_expert_utilization(expert_utilization)
            metrics['expert_analysis'] = expert_analysis
        
        # Generate evaluation report
        report = self._generate_evaluation_report(metrics, task)
        
        # Save predictions if requested
        if save_predictions and self.config.evaluation.get('save_predictions', True):
            self._save_predictions(all_predictions, all_targets, task)
        
        return {
            'metrics': metrics,
            'report': report,
            'predictions': all_predictions if save_predictions else None
        }
    
    def _compute_aggregate_metrics(self, all_outputs: List[Dict], 
                                 all_targets: List[Dict],
                                 task: str) -> Dict[str, float]:
        """Compute aggregate metrics across all batches"""
        # Concatenate all outputs and targets
        if task == 'link_prediction':
            all_scores = torch.cat([out['scores'] for out in all_outputs])
            all_labels = torch.cat([tgt['labels'] for tgt in all_targets])
            
            metrics = compute_metrics(
                {'scores': all_scores},
                {'labels': all_labels},
                task=task,
                config=self.config
            )
        
        elif task in ['entity_classification', 'relation_extraction']:
            all_logits = torch.cat([out['logits'] for out in all_outputs])
            all_labels = torch.cat([tgt['labels'] for tgt in all_targets])
            
            metrics = compute_metrics(
                {'logits': all_logits},
                {'labels': all_labels},
                task=task,
                config=self.config
            )
            
            # Add confusion matrix
            predictions = torch.argmax(all_logits, dim=-1).cpu().numpy()
            labels = all_labels.cpu().numpy()
            conf_matrix = confusion_matrix(labels, predictions)
            metrics['confusion_matrix'] = conf_matrix.tolist()
        
        return metrics
    
    def _analyze_expert_utilization(self, expert_utilization: List[torch.Tensor]) -> Dict[str, Any]:
        """Analyze how experts are being utilized"""
        # Stack all utilization scores
        utilization = torch.stack(expert_utilization)
        
        # Compute statistics
        mean_utilization = utilization.mean(dim=0).mean(dim=0)
        std_utilization = utilization.std(dim=0).mean(dim=0)
        
        # Expert activation frequency
        expert_names = ['action', 'spatial', 'temporal', 'semantic', 'safety']
        
        analysis = {
            'mean_utilization': {
                name: float(mean_utilization[i]) 
                for i, name in enumerate(expert_names)
            },
            'std_utilization': {
                name: float(std_utilization[i]) 
                for i, name in enumerate(expert_names)
            },
            'total_activations': utilization.sum().item()
        }
        
        return analysis
    
    def _generate_evaluation_report(self, metrics: Dict[str, Any], task: str) -> str:
        """Generate human-readable evaluation report"""
        report = f"Evaluation Report - Task: {task}\n"
        report += "=" * 50 + "\n\n"
        
        # Main metrics
        report += "Performance Metrics:\n"
        report += "-" * 30 + "\n"
        
        key_metrics = ['accuracy', 'f1', 'precision', 'recall', 'auc', 'mrr']
        for metric in key_metrics:
            if metric in metrics:
                report += f"{metric.upper():10s}: {metrics[metric]:.4f}\n"
        
        # Hits@K metrics
        if any('hits@' in k for k in metrics.keys()):
            report += "\nRanking Metrics:\n"
            report += "-" * 30 + "\n"
            for k in self.config.evaluation.get('k_values', [1, 3, 10]):
                if f'hits@{k}' in metrics:
                    report += f"Hits@{k:2d}    : {metrics[f'hits@{k}']:.4f}\n"
        
        # Expert analysis
        if 'expert_analysis' in metrics:
            report += "\nExpert Utilization:\n"
            report += "-" * 30 + "\n"
            for expert, util in metrics['expert_analysis']['mean_utilization'].items():
                report += f"{expert:10s}: {util:.3f} ± {metrics['expert_analysis']['std_utilization'][expert]:.3f}\n"
        
        return report
    
    def _save_predictions(self, predictions: List[Dict], targets: List[Dict], task: str):
        """Save predictions to file"""
        save_dir = Path(self.config.paths['model_dir']) / 'predictions'
        save_dir.mkdir(exist_ok=True)
        
        # Convert to serializable format
        saved_data = {
            'task': task,
            'predictions': [],
            'targets': []
        }
        
        for pred, tgt in zip(predictions, targets):
            saved_data['predictions'].append({
                k: v.tolist() if isinstance(v, torch.Tensor) else v 
                for k, v in pred.items()
            })
            saved_data['targets'].append({
                k: v.tolist() if isinstance(v, torch.Tensor) else v 
                for k, v in tgt.items() if k in ['labels', 'head', 'tail', 'relation']
            })
        
        # Save to JSON
        save_path = save_dir / f'{task}_predictions.json'
        with open(save_path, 'w') as f:
            json.dump(saved_data, f, indent=2)
        
        print(f"Predictions saved to {save_path}")
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                moved_batch[key] = self._move_batch_to_device(value)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def visualize_results(self, metrics: Dict[str, Any], save_path: Optional[str] = None):
        """Visualize evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance metrics bar plot
        ax = axes[0, 0]
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        ax.bar(metric_names, metric_values)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics')
        ax.set_ylabel('Score')
        
        # Expert utilization
        if 'expert_analysis' in metrics:
            ax = axes[0, 1]
            experts = list(metrics['expert_analysis']['mean_utilization'].keys())
            utilization = list(metrics['expert_analysis']['mean_utilization'].values())
            ax.bar(experts, utilization)
            ax.set_title('Expert Utilization')
            ax.set_ylabel('Mean Activation')
            ax.set_xticklabels(experts, rotation=45)
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            ax = axes[1, 0]
            conf_matrix = np.array(metrics['confusion_matrix'])
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        # Ranking metrics
        ax = axes[1, 1]
        k_values = self.config.evaluation.get('k_values', [1, 3, 10])
        hits_values = [metrics.get(f'hits@{k}', 0) for k in k_values]
        ax.plot(k_values, hits_values, 'bo-')
        ax.set_xlabel('K')
        ax.set_ylabel('Hits@K')
        ax.set_title('Ranking Performance')
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

    def visualize_predictions(self, predictions: Dict, save_dir: str):
        """可视化模型预测结果"""
        from utils.kg_visualization import KnowledgeGraphVisualizer

        visualizer = KnowledgeGraphVisualizer(self.config)

        # 将预测结果整合到图中
        enhanced_graph = self._enhance_graph_with_predictions(predictions)

        # 创建可视化
        net = visualizer.create_from_model_output(
            graph=enhanced_graph['graph'],
            entities=enhanced_graph['entities'],
            relations=enhanced_graph['relations'],
            predictions=enhanced_graph['confidence_scores']
        )

        # 保存不同视角的可视化
        save_path = Path(save_dir) / "prediction_visualization.html"
        visualizer.save_with_legend(net, str(save_path))