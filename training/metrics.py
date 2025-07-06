import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                           roc_auc_score, average_precision_score)
from typing import Dict, List, Optional, Tuple, Any

class Metrics:
    """Metrics computation for knowledge graph tasks"""
    
    def __init__(self, config):
        self.config = config
        self.k_values = config.evaluation.get('k_values', [1, 3, 10])
    
    def compute_link_prediction_metrics(self, 
                                      scores: torch.Tensor,
                                      labels: torch.Tensor,
                                      all_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute metrics for link prediction"""
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Binary predictions
        predictions = (scores_np > 0).astype(int)
        
        # Basic metrics
        accuracy = accuracy_score(labels_np, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, predictions, average='binary'
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # AUC and AP if scores are probabilities
        try:
            probs = torch.sigmoid(scores).detach().cpu().numpy()
            auc = roc_auc_score(labels_np, probs)
            ap = average_precision_score(labels_np, probs)
            metrics['auc'] = auc
            metrics['ap'] = ap
        except:
            pass
        
        # Ranking metrics if all scores provided
        if all_scores is not None:
            ranking_metrics = self.compute_ranking_metrics(all_scores, labels)
            metrics.update(ranking_metrics)
        
        return metrics
    
    def compute_ranking_metrics(self, scores: torch.Tensor, 
                              labels: torch.Tensor) -> Dict[str, float]:
        """Compute ranking metrics (MRR, Hits@K)"""
        # Sort scores in descending order
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)
        
        # Find positions of positive samples
        positive_mask = labels == 1
        positive_indices = positive_mask.nonzero(as_tuple=True)[0]
        
        ranks = []
        for idx in positive_indices:
            rank = (sorted_indices == idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
        
        if not ranks:
            return {}
        
        ranks = np.array(ranks)
        
        # MRR
        mrr = np.mean(1.0 / ranks)
        
        # Hits@K
        hits_at_k = {}
        for k in self.k_values:
            hits_at_k[f'hits@{k}'] = np.mean(ranks <= k)
        
        return {'mrr': mrr, **hits_at_k}
    
    def compute_classification_metrics(self, 
                                     logits: torch.Tensor,
                                     labels: torch.Tensor,
                                     num_classes: Optional[int] = None) -> Dict[str, float]:
        """Compute metrics for classification"""
        predictions = torch.argmax(logits, dim=-1)
        
        predictions_np = predictions.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Overall accuracy
        accuracy = accuracy_score(labels_np, predictions_np)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_np, predictions_np, average=None
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels_np, predictions_np, average='macro'
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels_np, predictions_np, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }
        
        # Add per-class metrics
        if num_classes:
            for i in range(min(num_classes, len(precision))):
                metrics[f'class_{i}_precision'] = precision[i]
                metrics[f'class_{i}_recall'] = recall[i]
                metrics[f'class_{i}_f1'] = f1[i]
        
        return metrics

def compute_metrics(outputs: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    task: str,
                    config: Optional[Any] = None) -> Dict[str, float]:
    """Compute metrics based on task"""
    metrics_computer = Metrics(config) if config else Metrics(type('Config', (), {'evaluation': {}})())
        
    # 处理标签字段的不同命名方式
    def get_labels(data):
        """从数据中提取标签，支持多种命名方式"""
        if hasattr(data, 'label') and data.label.numel() > 0:
            return data.label
        elif hasattr(data, 'labels') and data.labels.numel() > 0:
            return data.labels
        elif hasattr(data, 'edge_label') and data.edge_label.numel() > 0:
            return data.edge_label
        elif isinstance(data, dict):
            if 'label' in data and data['label'].numel() > 0:
                return data['label']
            elif 'labels' in data and data['labels'].numel() > 0:
                return data['labels']
            elif 'edge_label' in data and data['edge_label'].numel() > 0:
                return data['edge_label']
            elif 'y' in data and data['y'].numel() > 0:
                return data['y']
        return None
        
    labels = get_labels(targets)
     # 检查是否有有效的预测和标签
    if task == 'link_prediction':
        if 'scores' in outputs and outputs['scores'].numel() > 0 and labels is not None and labels.numel() > 0:
            scores = outputs['scores']
            if len(labels) != len(scores):
                min_len = min(len(labels), len(scores))
                labels = labels[:min_len]
                scores = scores[:min_len]
            
            print(f"[Metrics Debug] Computing metrics with {len(scores)} samples")
            return metrics_computer.compute_link_prediction_metrics(
                scores,
                labels,
                outputs.get('all_scores', None)
            )
        else:
            # 返回默认的指标值
            print(f"[Metrics Debug] No valid data for link prediction metrics")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    elif task == 'entity_classification':
        if 'logits' in outputs and outputs['logits'].numel() > 0 and labels is not None and labels.numel() > 0:
            return metrics_computer.compute_classification_metrics(
                outputs['logits'],
                labels
            )
        else:
            # 返回默认的指标值
            return {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0
            }
    elif task == 'relation_extraction':
        if 'logits' in outputs and outputs['logits'].numel() > 0 and labels is not None and labels.numel() > 0:
            return metrics_computer.compute_classification_metrics(
                outputs['logits'],
                labels
            )
        else:
            # 返回默认的指标值
            return {
                'accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0
            }
    else:
        return {}