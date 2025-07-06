import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

class MultiTaskLoss(nn.Module):
    """Multi-task loss for knowledge graph construction"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Task weights
        self.task_weights = {
            'link_prediction': 1.0,
            'entity_classification': 0.5,
            'relation_extraction': 0.8
        }

        # Initialize task-specific losses
        self.link_prediction_loss = nn.BCEWithLogitsLoss()
        self.entity_classification_loss = nn.CrossEntropyLoss()
        self.relation_extraction_loss = nn.CrossEntropyLoss()

        # Auxiliary losses
        self.contrastive_loss = ContrastiveLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                task: str) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        
        # 确定主设备（使用第一个输出张量的设备）
        main_device = next(iter(outputs.values())).device if outputs else torch.device('cpu')

        # Task-specific loss
        if task == 'link_prediction':
            if 'scores' in outputs and outputs['scores'].numel() > 0:
                # 处理标签字段的不同命名方式
                labels = None

                if hasattr(targets, 'label') and targets.label.numel() > 0:
                    labels = targets.label
                elif hasattr(targets, 'labels') and targets.labels.numel() > 0:
                    labels = targets.labels
                elif hasattr(targets, 'edge_label') and targets.edge_label.numel() > 0:
                    labels = targets.edge_label
                elif isinstance(targets, dict):
                    if 'label' in targets and targets['label'].numel() > 0:
                        labels = targets['label']
                    elif 'labels' in targets and targets['labels'].numel() > 0:
                        labels = targets['labels']
                    elif 'edge_label' in targets and targets['edge_label'].numel() > 0:
                        labels = targets['edge_label']
                    elif 'y' in targets and targets['y'].numel() > 0:
                        labels = targets['y']
                
                if labels is not None and labels.numel() > 0:
                    # 确保标签和预测的长度匹配
                    scores = outputs['scores']
                    if len(labels) != len(scores):
                        min_len = min(len(labels), len(scores))
                        labels = labels[:min_len]
                        scores = scores[:min_len]
                    
                    # 确保标签在正确的设备上
                    labels = labels.to(scores.device).float()

                     # 添加调试信息
                    print(f"[Loss Debug] scores shape: {scores.shape}, labels shape: {labels.shape}")
                    print(f"[Loss Debug] scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                    print(f"[Loss Debug] labels range: [{labels.min().item():.4f}, {labels.max().item():.4f}]")

                    task_loss = self.link_prediction_loss(scores, labels)
                    losses['link_prediction_loss'] = task_loss.to(main_device)
                    print(f"[Loss Debug] computed loss: {task_loss.item():.6f}")
                else:
                    print(f"[Loss Debug] No valid labels found for link prediction")
                    losses['link_prediction_loss'] = torch.tensor(0.0, device=main_device, requires_grad=True)
            else:
                # 如果没有有效的预测，创建一个零损失
                print(f"[Loss Debug] No valid scores found for link prediction")
                losses['link_prediction_loss'] = torch.tensor(0.0, device=main_device, requires_grad=True)

        elif task == 'entity_classification':
            if 'logits' in outputs and outputs['logits'].numel() > 0:
                # 处理标签字段的不同命名方式
                labels = None
                if hasattr(targets, 'label'):
                    labels = targets.label
                elif hasattr(targets, 'labels'):
                    labels = targets.labels
                elif isinstance(targets, dict):
                    if 'label' in targets:
                        labels = targets['label']
                    elif 'labels' in targets:
                        labels = targets['labels']
                    elif 'y' in targets:
                        labels = targets['y']
                
                if labels is not None and labels.numel() > 0:
                    # 确保标签在正确的设备上
                    labels = labels.to(outputs['logits'].device)
                    task_loss = self.entity_classification_loss(outputs['logits'], labels)
                    losses['entity_classification_loss'] = task_loss.to(main_device)

                    # Add type classification loss if available
                    if 'type_logits' in outputs and 'type_labels' in targets:
                        type_labels = targets['type_labels'].to(outputs['type_logits'].device)
                        type_loss = self.entity_classification_loss(outputs['type_logits'], type_labels)
                        losses['type_classification_loss'] = (type_loss * 0.5).to(main_device)
                else:
                    # 如果没有有效的标签，创建一个零损失
                    losses['entity_classification_loss'] = torch.tensor(0.0, device=main_device, requires_grad=True)
            else:
                # 如果没有有效的预测，创建一个零损失
                losses['entity_classification_loss'] = torch.tensor(0.0, device=main_device, requires_grad=True)

        elif task == 'relation_extraction':
            if 'logits' in outputs and outputs['logits'].numel() > 0:
                # 处理标签字段的不同命名方式
                labels = None
                if hasattr(targets, 'label'):
                    labels = targets.label
                elif hasattr(targets, 'labels'):
                    labels = targets.labels
                elif isinstance(targets, dict):
                    if 'label' in targets:
                        labels = targets['label']
                    elif 'labels' in targets:
                        labels = targets['labels']
                    elif 'y' in targets:
                        labels = targets['y']
                
                if labels is not None and labels.numel() > 0:
                    # 确保标签在正确的设备上
                    labels = labels.to(outputs['logits'].device)
                    task_loss = self.relation_extraction_loss(outputs['logits'], labels)
                    losses['relation_extraction_loss'] = task_loss.to(main_device)
                else:
                    # 如果没有有效的标签，创建一个零损失
                    losses['relation_extraction_loss'] = torch.tensor(0.0, device=main_device, requires_grad=True)
            else:
                # 如果没有有效的预测，创建一个零损失
                losses['relation_extraction_loss'] = torch.tensor(0.0, device=main_device, requires_grad=True)

        # Gating loss (load balancing) - 确保在主设备上
        if 'gating_loss' in outputs:
            gating_loss = outputs['gating_loss']
            if gating_loss.device != main_device:
                gating_loss = gating_loss.to(main_device)
            losses['gating_loss'] = gating_loss

        # Compute total loss - 现在所有损失都在同一设备上
        total_loss = sum(self.task_weights.get(k.replace('_loss', ''), 1.0) * v
                        for k, v in losses.items() if k != 'gating_loss')
        
        # 添加门控损失
        if 'gating_loss' in losses:
            total_loss = total_loss + 0.01 * losses['gating_loss']
        
        losses['total_loss'] = total_loss

        print(f"[Loss Debug] total_loss: {total_loss.item():.6f}")

        return losses

class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()

class ContrastiveLoss(nn.Module):
    """Contrastive loss for representation learning"""

    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negative: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss

        Args:
            anchor: Anchor embeddings [batch_size, dim]
            positive: Positive embeddings [batch_size, dim]
            negative: Negative embeddings [batch_size, dim] or None
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)

        if negative is None:
            # Use other samples in batch as negatives (SimCLR style)
            batch_size = anchor.size(0)

            # Compute similarity matrix
            sim_matrix = torch.matmul(anchor, positive.t()) / self.temperature

            # Mask out diagonal
            mask = torch.eye(batch_size, dtype=torch.bool, device=anchor.device)
            sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

            # Compute loss
            labels = torch.arange(batch_size, device=anchor.device)
            loss = F.cross_entropy(sim_matrix, labels)
        else:
            # Triplet loss style
            negative = F.normalize(negative, p=2, dim=1)

            pos_dist = torch.norm(anchor - positive, p=2, dim=1)
            neg_dist = torch.norm(anchor - negative, p=2, dim=1)

            loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        return loss

class RankingLoss(nn.Module):
    """Ranking loss for link prediction"""

    def __init__(self, margin: float = 1.0, num_neg_samples: int = 10):
        super().__init__()
        self.margin = margin
        self.num_neg_samples = num_neg_samples

    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute ranking loss

        Args:
            pos_scores: Positive sample scores [batch_size]
            neg_scores: Negative sample scores [batch_size, num_neg_samples]
        """
        # Expand positive scores
        pos_scores = pos_scores.unsqueeze(1).expand_as(neg_scores)

        # Compute margin ranking loss
        loss = F.relu(self.margin - pos_scores + neg_scores).mean()

        return loss