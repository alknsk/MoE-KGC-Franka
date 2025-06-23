import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple

from .encoders import TextEncoder, TabularEncoder, StructuredEncoder
from .experts import (ActionExpert, SpatialExpert, TemporalExpert,
                     SemanticExpert, SafetyExpert)
from .gating import AdaptiveGating
from .graph_layers import EnhancedGNN, GraphFusion
from .task_heads import (LinkPredictionHead, EntityClassificationHead,
                        RelationExtractionHead)

class MoEKGC(nn.Module):
    """
    Mixture of Experts model for Knowledge Graph Construction
    Specialized for Franka robot human-robot interaction
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize encoders
        self.text_encoder = TextEncoder(
            model_name="bert-base-uncased",
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.expert_hidden_dim,
            dropout_rate=config.model.dropout_rate
        )

        self.tabular_encoder = TabularEncoder(
            numerical_features=['joint_positions', 'gripper_state', 'force_torque'],
            categorical_features=['action', 'object_id'],
            embedding_dims={
                'action': {'vocab_size': config.data.vocab_size, 'embed_dim': 64},
                'object_id': {'vocab_size': 1000, 'embed_dim': 32}
            },
            hidden_dims=[config.model.expert_hidden_dim, config.model.expert_hidden_dim // 2],
            output_dim=config.model.expert_hidden_dim,
            dropout_rate=config.model.dropout_rate
        )

        self.structured_encoder = StructuredEncoder(
            input_dim=256,  # Placeholder
            hidden_dims=[config.model.expert_hidden_dim, config.model.expert_hidden_dim // 2],
            output_dim=config.model.expert_hidden_dim,
            dropout_rate=config.model.dropout_rate,
            use_graph_structure=True
        )

        # Initialize experts
        expert_input_dim = config.model.expert_hidden_dim
        self.experts = nn.ModuleDict({
            'action': ActionExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['action_expert']['hidden_dims'],
                output_dim=config.model.hidden_dim,
                dropout_rate=config.model.dropout_rate,
                use_attention=config.experts['action_expert']['use_attention'],
                num_joints=config.franka.joint_dim
            ),
            'spatial': SpatialExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['spatial_expert']['hidden_dims'],
                output_dim=config.model.hidden_dim,
                dropout_rate=config.model.dropout_rate,
                use_attention=config.experts['spatial_expert']['use_attention'],
                workspace_dim=3
            ),
            'temporal': TemporalExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['temporal_expert']['hidden_dims'],
                output_dim=config.model.hidden_dim,
                dropout_rate=config.model.dropout_rate,
                use_attention=config.experts['temporal_expert']['use_attention']
            ),
            'semantic': SemanticExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['semantic_expert']['hidden_dims'],
                output_dim=config.model.hidden_dim,
                dropout_rate=config.model.dropout_rate,
                use_attention=config.experts['semantic_expert']['use_attention'],
                vocab_size=config.data.vocab_size
            ),
            'safety': SafetyExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['safety_expert']['hidden_dims'],
                output_dim=config.model.hidden_dim,
                dropout_rate=config.model.dropout_rate,
                use_attention=config.experts['safety_expert']['use_attention']
            )
        })

        # Initialize gating mechanism
        self.gating = AdaptiveGating(
            input_dim=config.model.expert_hidden_dim * 3,  # 3 encoders
            num_experts=config.model.num_experts,
            hidden_dim=config.gating.temperature,
            temperature=config.gating.temperature,
            noise_std=config.gating.noise_std,
            top_k=config.gating.top_k,
            load_balancing_weight=config.gating.load_balancing_weight
        )

        # Initialize GNN
        self.gnn = EnhancedGNN(
            input_dim=config.model.hidden_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.hidden_dim,
            num_layers=config.graph.num_layers,
            edge_dim=config.graph.edge_hidden_dim if config.graph.use_edge_features else None,
            dropout=config.model.dropout_rate
        )

        # Initialize graph fusion
        self.graph_fusion = GraphFusion(
            input_dims=[config.model.hidden_dim] * config.model.num_experts,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.model.hidden_dim,
            fusion_type='attention',
            dropout=config.model.dropout_rate
        )

        # Initialize task heads
        self.link_prediction_head = LinkPredictionHead(
            entity_dim=config.model.hidden_dim,
            relation_dim=config.model.hidden_dim // 2,
            hidden_dim=config.model.hidden_dim,
            num_relations=config.data.num_relations,
            dropout=config.model.dropout_rate
        )

        self.entity_classification_head = EntityClassificationHead(
            input_dim=config.model.hidden_dim,
            num_classes=config.data.num_entity_types,
            hidden_dims=[config.model.hidden_dim, config.model.hidden_dim // 2],
            dropout=config.model.dropout_rate
        )

        self.relation_extraction_head = RelationExtractionHead(
            entity_dim=config.model.hidden_dim,
            num_relations=config.data.num_relations,
            hidden_dim=config.model.hidden_dim,
            dropout=config.model.dropout_rate
        )

    def encode_multimodal_input(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode different modalities of input data"""
        encoded = {}

        # Encode text data if available
        if 'text_inputs' in batch:
            text_encoded = self.text_encoder(
                batch['text_inputs']['input_ids'],
                batch['text_inputs']['attention_mask']
            )
            encoded['text'] = text_encoded

        # Encode tabular data if available
        if 'tabular_inputs' in batch:
            tabular_encoded = self.tabular_encoder(
                batch['tabular_inputs']['numerical'],
                batch['tabular_inputs']['categorical']
            )
            encoded['tabular'] = tabular_encoded

        # Encode structured data if available
        if 'structured_inputs' in batch:
            structured_encoded = self.structured_encoder(
                batch['structured_inputs']['task_features'],
                batch['structured_inputs']['constraint_features'],
                batch['structured_inputs']['safety_features']
            )
            encoded['structured'] = structured_encoded

        return encoded

    def apply_experts(self, encoded_features: torch.Tensor,
                     expert_indices: torch.Tensor,
                     expert_gates: torch.Tensor,
                     batch: Dict[str, Any]) -> torch.Tensor:
        """Apply selected experts to encoded features"""
        batch_size = encoded_features.size(0)
        output_dim = self.config.model.hidden_dim

        # Initialize output tensor
        expert_outputs = torch.zeros(batch_size, output_dim, device=encoded_features.device)

        # Apply each expert
        for i in range(batch_size):
            for j, (expert_idx, gate) in enumerate(zip(expert_indices[i], expert_gates[i])):
                expert_name = list(self.experts.keys())[expert_idx]
                expert = self.experts[expert_name]

                # Prepare expert-specific inputs
                expert_kwargs = {}
                if expert_name == 'action' and 'joint_positions' in batch:
                    expert_kwargs['joint_positions'] = batch['joint_positions'][i:i+1]
                elif expert_name == 'spatial' and 'positions' in batch:
                    expert_kwargs['positions'] = batch['positions'][i:i+1]
                elif expert_name == 'temporal' and 'timestamps' in batch:
                    expert_kwargs['timestamps'] = batch['timestamps'][i:i+1]

                # Apply expert
                expert_output = expert(encoded_features[i:i+1], **expert_kwargs)

                # Weight by gate
                expert_outputs[i] += gate * expert_output.squeeze(0)

        return expert_outputs

    def forward(self, batch: Dict[str, Any], task: str = 'link_prediction') -> Dict[str, torch.Tensor]:
        """
        Forward pass of MoE-KGC model

        Args:
            batch: Batch of multimodal data
            task: Target task ('link_prediction', 'entity_classification', 'relation_extraction')

        Returns:
            Task-specific outputs
        """
        # Encode multimodal inputs
        encoded = self.encode_multimodal_input(batch)

        # Combine encoded features
        combined_features = []
        for modality in ['text', 'tabular', 'structured']:
            if modality in encoded:
                combined_features.append(encoded[modality])

        if combined_features:
            combined_features = torch.cat(combined_features, dim=-1)
        else:
            raise ValueError("No input modalities found in batch")

        # Apply gating mechanism
        gating_output = self.gating(combined_features)
        expert_indices = gating_output['indices']
        expert_gates = gating_output['gates']

        # Apply selected experts
        expert_outputs = self.apply_experts(
            combined_features, expert_indices, expert_gates, batch
        )

        # Apply GNN if graph structure is available
        if 'edge_index' in batch:
            gnn_output = self.gnn(
                expert_outputs,
                batch['edge_index'],
                batch.get('edge_attr', None),
                batch.get('batch_idx', None)
            )
            node_embeddings = gnn_output['node_embeddings']
            graph_embedding = gnn_output['graph_embedding']
        else:
            node_embeddings = expert_outputs
            graph_embedding = expert_outputs.mean(dim=0, keepdim=True)

        # Task-specific heads
        if task == 'link_prediction':
            output = self.link_prediction_head(
                batch['head_embeddings'],
                batch['tail_embeddings'],
                batch['relation_ids']
            )
        elif task == 'entity_classification':
            output = self.entity_classification_head(
                node_embeddings,
                batch.get('node_mask', None)
            )
        elif task == 'relation_extraction':
            output = self.relation_extraction_head(
                batch['head_embeddings'],
                batch['tail_embeddings'],
                batch.get('context', None)
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        # Add auxiliary outputs
        output['gating_loss'] = gating_output['load_balancing_loss']
        output['expert_utilization'] = gating_output.get('all_scores', None)

        return output

    def get_expert_weights(self) -> Dict[str, torch.Tensor]:
        """Get current expert importance weights"""
        return {
            name: expert.state_dict()
            for name, expert in self.experts.items()
        }

    def freeze_encoders(self):
        """Freeze encoder parameters"""
        for encoder in [self.text_encoder, self.tabular_encoder, self.structured_encoder]:
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze encoder parameters"""
        for encoder in [self.text_encoder, self.tabular_encoder, self.structured_encoder]:
            for param in encoder.parameters():
                param.requires_grad = True