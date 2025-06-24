import torch
import numpy as np
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import hashlib
import pickle


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device

    Args:
        device: Device string ('cuda', 'cpu', or None for auto)

    Returns:
        torch.device object
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'

    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_json(data: Dict[str, Any], path: Union[str, Path]):
    """
    Save dictionary to JSON file

    Args:
        data: Data to save
        path: Save path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dictionary from JSON file

    Args:
        path: JSON file path

    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: Union[str, Path]):
    """
    Save data to pickle file

    Args:
        data: Data to save
        path: Save path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load data from pickle file

    Args:
        path: Pickle file path

    Returns:
        Loaded data
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_hash(data: str) -> str:
    """
    Compute SHA256 hash of string

    Args:
        data: Input string

    Returns:
        Hash string
    """
    return hashlib.sha256(data.encode()).hexdigest()


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary

    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for keys

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    """
    Unflatten dictionary

    Args:
        d: Flattened dictionary
        sep: Separator used in keys

    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move batch to device

    Args:
        batch: Batch dictionary
        device: Target device

    Returns:
        Batch on device
    """
    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device)
        elif isinstance(value, dict):
            moved_batch[key] = batch_to_device(value, device)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            moved_batch[key] = [v.to(device) for v in value]
        else:
            moved_batch[key] = value
    return moved_batch


def create_mask(lengths: List[int], max_length: Optional[int] = None) -> torch.Tensor:
    """
    Create attention mask from sequence lengths

    Args:
        lengths: List of sequence lengths
        max_length: Maximum sequence length

    Returns:
        Boolean mask tensor
    """
    if max_length is None:
        max_length = max(lengths)

    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return mask


def get_activation_function(name: str) -> torch.nn.Module:
    """
    Get activation function by name

    Args:
        name: Activation function name

    Returns:
        Activation function module
    """
    activations = {
        'relu': torch.nn.ReLU(),
        'gelu': torch.nn.GELU(),
        'tanh': torch.nn.Tanh(),
        'sigmoid': torch.nn.Sigmoid(),
        'leaky_relu': torch.nn.LeakyReLU(),
        'elu': torch.nn.ELU(),
        'selu': torch.nn.SELU(),
        'swish': torch.nn.SiLU()
    }

    if name.lower() not in activations:
        raise ValueError(f"Unknown activation function: {name}")

    return activations[name.lower()]


def get_optimizer(name: str, parameters, lr: float, **kwargs) -> torch.optim.Optimizer:
    """
    Get optimizer by name

    Args:
        name: Optimizer name
        parameters: Model parameters
        lr: Learning rate
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop,
        'adagrad': torch.optim.Adagrad
    }

    if name.lower() not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}")

    return optimizers[name.lower()](parameters, lr=lr, **kwargs)


def get_scheduler(name: str, optimizer, **kwargs):
    """
    Get learning rate scheduler by name

    Args:
        name: Scheduler name
        optimizer: Optimizer instance
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    schedulers = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'step': torch.optim.lr_scheduler.StepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'linear': torch.optim.lr_scheduler.LinearLR,
        'onecycle': torch.optim.lr_scheduler.OneCycleLR
    }

    if name.lower() not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}")

    return schedulers[name.lower()](optimizer, **kwargs)


class EarlyStopping:
    """Early stopping helper"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta