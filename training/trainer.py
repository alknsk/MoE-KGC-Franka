import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional, List, Tuple, Any
import numpy as np
from tqdm import tqdm
import wandb
import os
from pathlib import Path

from .losses import MultiTaskLoss
from .metrics import Metrics, compute_metrics

class Trainer:
    """Trainer for MoE-KGC model"""
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: torch.device = None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model.to(self.device)
        self.use_amp = getattr(config.training, 'mixed_precision', False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        print("weight_decay:", config.training.weight_decay, type(config.training.weight_decay))
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=float(config.training.weight_decay)
        )
        
        # Initialize scheduler
        if config.training.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.epochs,
                eta_min=config.training.learning_rate * 0.01
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        
        # Initialize loss
        self.criterion = MultiTaskLoss(config)
        
        # Initialize metrics
        self.metrics = Metrics(config)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_score = -float('inf')
        
        # Initialize wandb if configured
        if hasattr(config, 'wandb') and config.wandb.get('use_wandb', False):
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.run_name,
                config=config
            )
    
    def train_epoch(self, train_loader: DataLoader, task: str = 'link_prediction') -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = []
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 处理PyG batch
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)
            else:
                batch = self._move_batch_to_device(batch)
            
            # 梯度累积
            is_accumulation_step = (batch_idx + 1) % getattr(self.config.training, 'accumulation_steps', 1) != 0
            
            # 混合精度训练
            with torch.amp.autocast('cuda', enabled = getattr(self.config.training,'mixed_precision', False)):
                # 前向传播
                outputs = self.model(batch, task=task)
                
                # 计算损失
                loss_dict = self.criterion(outputs, batch, task=task)
                total_loss = loss_dict['total_loss']
                
                # 添加门控损失
                # if 'gating_loss' in outputs:
                #    gating_weight = getattr(self.config.training, 'gating_loss_weight', 0.01)
                #    total_loss = total_loss + gating_weight * outputs['gating_loss']
            
            # 梯度累积归一化
            if getattr(self.config.training, 'accumulation_steps', 1) > 1:
                total_loss = total_loss / getattr(self.config.training, 'accumulation_steps', 1)
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            if not is_accumulation_step:
                if self.config.training.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.training.gradient_clip
                    )
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
            
            # 更新指标
            epoch_losses.append(total_loss.item() * getattr(self.config.training, 'accumulation_steps', 1))
            
            with torch.no_grad():
                # 确保outputs和batch都在同一设备上
                metrics = compute_metrics(outputs, batch, task=task)
                epoch_metrics.append(metrics)
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'acc': f"{metrics.get('accuracy', 0):.4f}",
                'nodes': batch.num_nodes if hasattr(batch, 'num_nodes') else 'N/A'
            })
            
            # 定期清理GPU内存
            if batch_idx % getattr(self.config.training, 'empty_cache_freq', 10) == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.global_step += 1
        
        # 聚合epoch指标
        avg_loss = np.mean(epoch_losses)
        avg_metrics = self._aggregate_metrics(epoch_metrics)
        
        return {'loss': avg_loss, **avg_metrics}
    
    def validate(self, val_loader: DataLoader, task: str = 'link_prediction') -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch, task=task)
                
                # Compute loss
                loss_dict = self.criterion(outputs, batch, task=task)
                val_losses.append(loss_dict['total_loss'].item())
                
                # Compute metrics
                metrics = compute_metrics(outputs, batch, task=task)
                val_metrics.append(metrics)
        
        # Aggregate validation metrics
        avg_loss = np.mean(val_losses)
        avg_metrics = self._aggregate_metrics(val_metrics)
        
        return {'loss': avg_loss, **avg_metrics}
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              num_epochs: Optional[int] = None,
              task: str = 'link_prediction',
              save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """Full training loop"""
        num_epochs = num_epochs or self.config.training.epochs
        save_dir = Path(save_dir or self.config.paths['model_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_results = self.train_epoch(train_loader, task=task)
            history['train_loss'].append(train_results['loss'])
            history['train_metrics'].append(train_results)
            
            # Validation
            if val_loader is not None:
                val_results = self.validate(val_loader, task=task)
                history['val_loss'].append(val_results['loss'])
                history['val_metrics'].append(val_results)
                
                # Learning rate scheduling
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_results['loss'])
                else:
                    self.scheduler.step()
                
                # Check for improvement
                val_score = val_results.get('f1', val_results.get('accuracy', 0))
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    self.save_checkpoint(save_dir / 'best_model.pt')
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log({
                        'epoch': epoch,
                        'train/epoch_loss': train_results['loss'],
                        'val/epoch_loss': val_results['loss'],
                        **{f'train/{k}': v for k, v in train_results.items() if k != 'loss'},
                        **{f'val/{k}': v for k, v in val_results.items() if k != 'loss'}
                    })
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(save_dir / f'checkpoint_epoch_{epoch+1}.pt')
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_results['loss']:.4f}")
            if val_loader is not None:
                print(f"Val Loss: {val_results['loss']:.4f}")
                print(f"Best Val Score: {self.best_val_score:.4f}")
        
        # Save final model
        self.save_checkpoint(save_dir / 'final_model.pt')
        # 如果没有best_model.pt，则复制一份
        import shutil
        best_model_path = save_dir / 'best_model.pt'
        if not best_model_path.exists():
            shutil.copy(save_dir / 'final_model.pt', best_model_path)
        return history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_score = checkpoint['best_val_score']
        
        print(f"Checkpoint loaded from {path}")
    
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
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics over batches"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = np.mean(values)
        
        return aggregated