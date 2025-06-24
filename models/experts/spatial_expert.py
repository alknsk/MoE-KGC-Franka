import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from .base_expert import BaseExpert

class SpatialExpert(BaseExpert):
    """Expert for spatial reasoning and relationships"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 output_dim: int = 256,
                 dropout_rate: float = 0.1,
                 use_attention: bool = True,
                 workspace_dim: int = 3,
                 num_spatial_relations: int = 12):
        super().__init__(input_dim, hidden_dims, output_dim, dropout_rate, use_attention)
        
        self.workspace_dim = workspace_dim
        self.num_spatial_relations = num_spatial_relations
        
        # Position encoder
        self.position_encoder = nn.Sequential(
            nn.Linear(workspace_dim, hidden_dims[0] // 3),
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 3, hidden_dims[0] // 3)
        )
        
        # Orientation encoder (quaternion or euler angles)
        self.orientation_encoder = nn.Sequential(
            nn.Linear(4, hidden_dims[0] // 3),  # Quaternion input
            nn.ReLU(),
            nn.Linear(hidden_dims[0] // 3, hidden_dims[0] // 3)
        )
        
        # Spatial relation classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[1], num_spatial_relations)
        )
        
        # 3D convolution for voxel grid processing
        self.voxel_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        
        # Spatial transformer network components
        self.localization = nn.Sequential(
            nn.Linear(hidden_dims[0], 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 4)  # 3x4 affine transformation matrix
        )
    
    def compute_expert_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute spatial features"""
        features = []
        
        # Process positions
        if 'positions' in kwargs:
            positions = kwargs['positions']
            pos_features = self.position_encoder(positions)
            features.append(pos_features)
        
        # Process orientations
        if 'orientations' in kwargs:
            orientations = kwargs['orientations']
            orient_features = self.orientation_encoder(orientations)
            features.append(orient_features)
        
        # Process voxel grids
        if 'voxel_grid' in kwargs:
            voxel_grid = kwargs['voxel_grid']
            voxel_features = self.voxel_conv(voxel_grid.unsqueeze(1))
            voxel_features = voxel_features.squeeze(-1).squeeze(-1).squeeze(-1)
            features.append(voxel_features)
        
        # If no specific features, use input directly
        if not features:
            return x
        
        # Combine features
        if len(features) == 1:
            return features[0]
        else:
            return torch.cat(features, dim=-1)
    
    def compute_spatial_relation(self, obj1_features: torch.Tensor, 
                               obj2_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute spatial relation between two objects"""
        # Encode both objects
        enc1 = self.forward(obj1_features)
        enc2 = self.forward(obj2_features)
        
        # Concatenate encodings
        combined = torch.cat([enc1, enc2], dim=-1)
        
        # Classify relation
        relation_logits = self.relation_classifier(combined)
        relation_probs = torch.softmax(relation_logits, dim=-1)
        
        return {
            'logits': relation_logits,
            'probabilities': relation_probs,
            'predicted_relation': torch.argmax(relation_probs, dim=-1)
        }
    
    def transform_coordinates(self, positions: torch.Tensor, 
                            reference_frame: torch.Tensor) -> torch.Tensor:
        """Transform coordinates to different reference frame"""
        batch_size = positions.size(0)
        
        # Compute transformation matrix
        theta = self.localization(reference_frame)
        theta = theta.view(-1, 3, 4)
        
        # Add homogeneous coordinate
        ones = torch.ones(batch_size, positions.size(1), 1, device=positions.device)
        positions_homo = torch.cat([positions, ones], dim=-1)
        
        # Apply transformation
        transformed = torch.bmm(positions_homo, theta.transpose(1, 2))
        
        return transformed
    
    def compute_workspace_occupancy(self, positions: torch.Tensor, 
                                  workspace_bounds: Dict[str, Tuple[float, float]],
                                  grid_resolution: int = 32) -> torch.Tensor:
        """Compute workspace occupancy grid"""
        batch_size = positions.size(0)
        
        # Create empty voxel grid
        voxel_grid = torch.zeros(
            batch_size, grid_resolution, grid_resolution, grid_resolution,
            device=positions.device
        )
        
        # Normalize positions to grid coordinates
        for dim, (dim_name, (min_val, max_val)) in enumerate(workspace_bounds.items()):
            positions[:, :, dim] = (positions[:, :, dim] - min_val) / (max_val - min_val)
        
        # Convert to grid indices
        grid_indices = (positions * (grid_resolution - 1)).long()
        grid_indices = torch.clamp(grid_indices, 0, grid_resolution - 1)
        
        # Fill voxel grid
        for b in range(batch_size):
            for point_idx in range(grid_indices.size(1)):
                x, y, z = grid_indices[b, point_idx]
                voxel_grid[b, x, y, z] = 1.0
        
        return voxel_grid