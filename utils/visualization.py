import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path


def visualize_graph(graph: nx.Graph,
                    node_colors: Optional[Dict[str, str]] = None,
                    edge_colors: Optional[Dict[str, str]] = None,
                    title: str = "Knowledge Graph",
                    figsize: Tuple[int, int] = (12, 8),
                    save_path: Optional[str] = None,
                    layout: str = 'spring'):
    """
    Visualize a NetworkX graph

    Args:
        graph: NetworkX graph
        node_colors: Dict mapping node types to colors
        edge_colors: Dict mapping edge types to colors
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        layout: Graph layout algorithm
    """
    plt.figure(figsize=figsize)

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(graph, k=2, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.random_layout(graph)

    # Default colors
    if node_colors is None:
        node_colors = {
            'action': '#87CEEB',  # Sky blue
            'object': '#90EE90',  # Light green
            'task': '#FFB6C1',  # Light pink
            'constraint': '#FFFFE0',  # Light yellow
            'safety': '#DDA0DD',  # Plum
            'spatial': '#F0E68C',  # Khaki
            'temporal': '#D3D3D3'  # Light gray
        }

    # Draw nodes by type
    for node_type, color in node_colors.items():
        nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == node_type]
        if nodes:
            nx.draw_networkx_nodes(graph, pos, nodelist=nodes,
                                   node_color=color, node_size=500,
                                   label=node_type, alpha=0.8)

    # Draw edges
    if hasattr(graph, 'edges'):
        nx.draw_networkx_edges(graph, pos, edge_color='gray',
                               arrows=True, arrowsize=20, alpha=0.5)

    # Draw labels
    labels = {}
    for node, data in graph.nodes(data=True):
        label = data.get('name', str(node))
        if len(label) > 15:
            label = label[:12] + '...'
        labels[node] = label

    nx.draw_networkx_labels(graph, pos, labels, font_size=8)

    plt.title(title, fontsize=16)
    plt.legend(loc='upper right')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(history: Dict[str, List[float]],
                          title: str = "Training History",
                          save_path: Optional[str] = None):
    """
    Plot training history

    Args:
        history: Dict containing training metrics history
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # Loss plot
    ax = axes[0, 0]
    if 'train_loss' in history:
        ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[0, 1]
    if 'train_metrics' in history and history['train_metrics']:
        train_acc = [m.get('accuracy', 0) for m in history['train_metrics']]
        ax.plot(train_acc, label='Train Accuracy', linewidth=2)
    if 'val_metrics' in history and history['val_metrics']:
        val_acc = [m.get('accuracy', 0) for m in history['val_metrics']]
        ax.plot(val_acc, label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1 Score plot
    ax = axes[1, 0]
    if 'train_metrics' in history and history['train_metrics']:
        train_f1 = [m.get('f1', 0) for m in history['train_metrics']]
        ax.plot(train_f1, label='Train F1', linewidth=2)
    if 'val_metrics' in history and history['val_metrics']:
        val_f1 = [m.get('f1', 0) for m in history['val_metrics']]
        ax.plot(val_f1, label='Val F1', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot
    ax = axes[1, 1]
    if 'learning_rates' in history:
        ax.plot(history['learning_rates'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_expert_utilization(expert_scores: np.ndarray,
                            expert_names: List[str] = None,
                            title: str = "Expert Utilization",
                            save_path: Optional[str] = None):
    """
    Plot expert utilization heatmap

    Args:
        expert_scores: Expert activation scores [num_samples, num_experts]
        expert_names: List of expert names
        title: Plot title
        save_path: Path to save figure
    """
    if expert_names is None:
        expert_names = ['Action', 'Spatial', 'Temporal', 'Semantic', 'Safety']

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(expert_scores.T,
                xticklabels=False,
                yticklabels=expert_names,
                cmap='YlOrRd',
                cbar_kws={'label': 'Activation Score'})

    plt.xlabel('Samples')
    plt.ylabel('Experts')
    plt.title(title)

    # Add average utilization text
    avg_utilization = expert_scores.mean(axis=0)
    for i, (name, score) in enumerate(zip(expert_names, avg_utilization)):
        plt.text(expert_scores.shape[0] + 1, i + 0.5, f'{score:.3f}',
                 va='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_attention_weights(attention_weights: torch.Tensor,
                                source_tokens: List[str],
                                target_tokens: List[str],
                                title: str = "Attention Weights",
                                save_path: Optional[str] = None):
    """
    Visualize attention weights

    Args:
        attention_weights: Attention weight matrix
        source_tokens: List of source tokens
        target_tokens: List of target tokens
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 8))

    # Convert to numpy if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()

    # Create heatmap
    sns.heatmap(attention_weights,
                xticklabels=target_tokens,
                yticklabels=source_tokens,
                cmap='Blues',
                cbar_kws={'label': 'Attention Weight'})

    plt.xlabel('Target')
    plt.ylabel('Source')
    plt.title(title)

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_trajectory(positions: np.ndarray,
                       title: str = "3D Trajectory",
                       save_path: Optional[str] = None):
    """
    Plot 3D trajectory

    Args:
        positions: 3D positions [num_points, 3]
        title: Plot title
        save_path: Path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=2, alpha=0.7)

    # Mark start and end
    ax.scatter(*positions[0], color='green', s=100, label='Start')
    ax.scatter(*positions[-1], color='red', s=100, label='End')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()