import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple , Any
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch_geometric.nn import GATConv, GINConv, SAGEConv, RGCNConv

class BaselineModel(nn.Module):
    """Wrapper for baseline GNN models"""
    
    def __init__(self, model_type: str, input_dim: int, hidden_dim: int, 
                 output_dim: int, num_layers: int = 3, num_relations: int = None):
        super().__init__()
        self.model_type = model_type
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            
            if model_type == 'GAT':
                layer = GATConv(in_dim, out_dim // 8, heads=8, concat=(i < num_layers - 1))
            elif model_type == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim)
                )
                layer = GINConv(mlp)
            elif model_type == 'GraphSAGE':
                layer = SAGEConv(in_dim, out_dim)
            elif model_type == 'RGCN':
                layer = RGCNConv(in_dim, out_dim, num_relations or 50)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.layers.append(layer)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_type: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            if self.model_type == 'RGCN' and edge_type is not None:
                x = layer(x, edge_index, edge_type)
            else:
                x = layer(x, edge_index)
            
            if i < len(self.layers) - 1:
                x = torch.relu(x)
                x = self.dropout(x)
        
        return x

class BaselineComparison:
    """Compare MoE-KGC with baseline models"""
    
    def __init__(self, moe_model: nn.Module, config, device: torch.device = None):
        self.moe_model = moe_model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize baseline models
        self.baseline_models = self._initialize_baselines()
    
    def _initialize_baselines(self) -> Dict[str, nn.Module]:
        """Initialize baseline models"""
        baselines = {}
        
        model_configs = {
            'GAT': {'hidden_dim': 256},
            'GIN': {'hidden_dim': 256},
            'GraphSAGE': {'hidden_dim': 256},
            'RGCN': {'hidden_dim': 256, 'num_relations': self.config.data.num_relations}
        }
        
        for model_type, model_config in model_configs.items():
            model = BaselineModel(
                model_type=model_type,
                input_dim=self.config['model']['hidden_dim'],
                hidden_dim=model_config['hidden_dim'],
                output_dim=self.config['model']['hidden_dim'],
                num_layers=self.config.graph.num_layers,
                num_relations=model_config.get('num_relations')
            )
            model.to(self.device)
            baselines[model_type] = model
        
        return baselines
    
    def compare_models(self, test_loader: DataLoader, task: str = 'link_prediction') -> pd.DataFrame:
        """Compare all models on test set"""
        results = []
        
        # Evaluate MoE model
        print("Evaluating MoE-KGC model...")
        moe_results = self._evaluate_model(self.moe_model, test_loader, task, 'MoE-KGC')
        results.append(moe_results)
        
        # Evaluate baseline models
        for model_name, model in self.baseline_models.items():
            print(f"Evaluating {model_name} model...")
            baseline_results = self._evaluate_model(model, test_loader, task, model_name)
            results.append(baseline_results)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        
        # Add relative improvement
        baseline_f1 = comparison_df[comparison_df['model'] != 'MoE-KGC']['f1'].max()
        moe_f1 = comparison_df[comparison_df['model'] == 'MoE-KGC']['f1'].values[0]
        improvement = ((moe_f1 - baseline_f1) / baseline_f1) * 100
        
        print(f"\nMoE-KGC improvement over best baseline: {improvement:.2f}%")
        
        return comparison_df
    
    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                       task: str, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model"""
        model.eval()
        
        all_predictions = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Evaluating {model_name}'):
                batch = self._move_batch_to_device(batch)
                
                # Time inference
                start_time = time.time()
                
                if model_name == 'MoE-KGC':
                    outputs = model(batch, task=task)
                    if task == 'link_prediction':
                        predictions = outputs['scores']
                    else:
                        predictions = outputs['logits']
                else:
                    # For baseline models, we need to adapt the forward pass
                    x = batch['node_features']
                    edge_index = batch['edge_index']
                    edge_type = batch.get('edge_type', None)
                    
                    node_embeddings = model(x, edge_index, edge_type)
                    
                    # Simple task head for fair comparison
                    if task == 'link_prediction':
                        head_emb = node_embeddings[batch['head_indices']]
                        tail_emb = node_embeddings[batch['tail_indices']]
                        predictions = (head_emb * tail_emb).sum(dim=-1)
                    else:
                        predictions = node_embeddings
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(batch['labels'].cpu())
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        if task == 'link_prediction':
            predictions_binary = (all_predictions > 0).float()
            accuracy = (predictions_binary == all_labels).float().mean().item()
            
            # Compute other metrics
            from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels.numpy(), predictions_binary.numpy(), average='binary'
            )
            
            try:
                auc = roc_auc_score(all_labels.numpy(), torch.sigmoid(all_predictions).numpy())
            except:
                auc = 0.0
        else:
            # For classification tasks
            predictions_class = torch.argmax(all_predictions, dim=-1)
            accuracy = (predictions_class == all_labels).float().mean().item()
            
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels.numpy(), predictions_class.numpy(), average='macro'
            )
            auc = 0.0
        
        # Model complexity
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'avg_inference_time': np.mean(inference_times),
            'num_parameters': num_params
        }
    
    def visualize_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[str] = None):
        """Visualize model comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics comparison
        ax = axes[0, 0]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        comparison_df.set_index('model')[metrics].plot(kind='bar', ax=ax)
        ax.set_title('Performance Metrics Comparison')
        ax.set_ylabel('Score')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1)
        
        # AUC comparison
        ax = axes[0, 1]
        comparison_df.set_index('model')['auc'].plot(kind='bar', ax=ax, color='orange')
        ax.set_title('AUC Score Comparison')
        ax.set_ylabel('AUC')
        ax.set_ylim(0, 1)
        
        # Inference time comparison
        ax = axes[1, 0]
        comparison_df.set_index('model')['avg_inference_time'].plot(kind='bar', ax=ax, color='green')
        ax.set_title('Average Inference Time')
        ax.set_ylabel('Time (seconds)')
        
        # Model complexity
        ax = axes[1, 1]
        comparison_df.set_index('model')['num_parameters'].plot(kind='bar', ax=ax, color='red')
        ax.set_title('Model Complexity')
        ax.set_ylabel('Number of Parameters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")
        else:
            plt.show()
    
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
    
    def generate_comparison_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate comparison report"""
        report = "Model Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        # Best model for each metric
        report += "Best Model by Metric:\n"
        report += "-" * 30 + "\n"
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'model']
            best_score = comparison_df[metric].max()
            report += f"{metric:10s}: {best_model:15s} ({best_score:.4f})\n"
        
        # Efficiency analysis
        report += "\nEfficiency Analysis:\n"
        report += "-" * 30 + "\n"
        
        fastest_model = comparison_df.loc[comparison_df['avg_inference_time'].idxmin(), 'model']
        fastest_time = comparison_df['avg_inference_time'].min()
        report += f"Fastest:     {fastest_model:15s} ({fastest_time:.4f}s)\n"
        
        smallest_model = comparison_df.loc[comparison_df['num_parameters'].idxmin(), 'model']
        smallest_params = comparison_df['num_parameters'].min()
        report += f"Smallest:    {smallest_model:15s} ({smallest_params:,} params)\n"
        
        # MoE-KGC specific analysis
        moe_row = comparison_df[comparison_df['model'] == 'MoE-KGC'].iloc[0]
        report += "\nMoE-KGC Performance:\n"
        report += "-" * 30 + "\n"
        report += f"F1 Score:    {moe_row['f1']:.4f}\n"
        report += f"Parameters:  {moe_row['num_parameters']:,}\n"
        report += f"Inference:   {moe_row['avg_inference_time']:.4f}s\n"
        
        return report