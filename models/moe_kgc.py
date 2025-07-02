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
            hidden_dim=config.hidden_dim,#删去model
            output_dim=config.expert_hidden_dim,#删去model
            dropout_rate=config.dropout_rate#删去model
        )

        self.tabular_encoder = TabularEncoder(
            numerical_features=['joint_positions', 'gripper_state', 'force_torque'],
            categorical_features=['action', 'object_id'],
            embedding_dims={
                'action': {'vocab_size': config.data.vocab_size, 'embed_dim': 64},
                'object_id': {'vocab_size': 1000, 'embed_dim': 32}
            },
            hidden_dims=[config.expert_hidden_dim, config.expert_hidden_dim // 2],
            output_dim=config.expert_hidden_dim,
            dropout_rate=config.dropout_rate
        )

        self.structured_encoder = StructuredEncoder(
            input_dim=256,  # Placeholder
            hidden_dims=[config.expert_hidden_dim, config.expert_hidden_dim // 2],
            output_dim=config.expert_hidden_dim,
            dropout_rate=config.dropout_rate,
            use_graph_structure=True
        )

        # Initialize experts
        expert_input_dim = config.expert_hidden_dim
        self.experts = nn.ModuleDict({
            'action': ActionExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['action_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['action_expert'].use_attention,
                num_joints=config.franka.joint_dim
            ),
            'spatial': SpatialExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['spatial_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['spatial_expert'].use_attention,
                workspace_dim=3
            ),
            'temporal': TemporalExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['temporal_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['temporal_expert'].use_attention
            ),
            'semantic': SemanticExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['semantic_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['semantic_expert'].use_attention,
                vocab_size=config.data.vocab_size
            ),
            'safety': SafetyExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['safety_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['safety_expert'].use_attention
            )
        })

        # Initialize gating mechanism
        self.gating = AdaptiveGating(
            input_dim=config.expert_hidden_dim * 3,  # 3 encoders
            num_experts=config.num_experts,
            hidden_dim=config.gating.hidden_dim,
            temperature=config.gating.temperature,
            noise_std=config.gating.noise_std,
            top_k=config.gating.top_k,
            load_balancing_weight=config.gating.load_balancing_weight
        )

        # Initialize GNN
        self.gnn = EnhancedGNN(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.graph.num_layers,
            edge_dim=config.graph.edge_hidden_dim if config.graph.use_edge_features else None,
            dropout=config.dropout_rate
        )

        # Initialize graph fusion
        self.graph_fusion = GraphFusion(
            input_dims=[config.hidden_dim] * config.num_experts,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            fusion_type='attention',
            dropout=config.dropout_rate
        )

        # Initialize task heads
        self.link_prediction_head = LinkPredictionHead(
            entity_dim=config.hidden_dim,
            relation_dim=config.hidden_dim // 2,
            hidden_dim=config.hidden_dim,
            num_relations=config.data.num_relations,
            dropout=config.dropout_rate
        )

        self.entity_classification_head = EntityClassificationHead(
            input_dim=config.hidden_dim,
            num_classes=config.data.num_entity_types,
            hidden_dims=[config.hidden_dim, config.hidden_dim // 2],
            dropout=config.dropout_rate
        )

        self.relation_extraction_head = RelationExtractionHead(
            entity_dim=config.hidden_dim,
            num_relations=config.data.num_relations,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate
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
            print("text_encoded shape:", text_encoded.shape)  # Debugging line
            encoded['text'] = text_encoded

        # Encode tabular data if available
        if 'tabular_inputs' in batch:
            tabular_encoded = self.tabular_encoder(
                batch['tabular_inputs']['numerical'],
                batch['tabular_inputs']['categorical']
            )
            print("tabular_encoded shape:", tabular_encoded.shape) # Debugging line
            encoded['tabular'] = tabular_encoded

        elif 'node_features' in batch:
            # 只有没有tabular_inputs时才用node_features
            print("node_features shape:", batch['node_features'].shape)
            # 这里建议只在特殊情况用，并确保shape为(batch, D)
            encoded['tabular'] = batch['node_features']
        
        # Encode structured data if available
        if 'structured_inputs' in batch:
            structured_encoded = self.structured_encoder(
                batch['structured_inputs']['task_features'],
                batch['structured_inputs']['constraint_features'],
                batch['structured_inputs']['safety_features']
            )
            print("structured_encoded shape:", structured_encoded.shape) # Debugging line
            encoded['structured'] = structured_encoded
            
        for k in ['text', 'tabular', 'structured']:
            if k in encoded:
                assert encoded[k].shape[1] == self.config.expert_hidden_dim, \
                    f"{k} shape[1] ({encoded[k].shape[1]}) != expert_hidden_dim ({self.config.expert_hidden_dim})"
        
        return encoded

    def apply_experts(self, expert_inputs: Dict[str, torch.Tensor],
                     expert_indices: torch.Tensor,
                     expert_gates: torch.Tensor,
                     batch: Dict[str, Any]) -> torch.Tensor:
        """Apply selected experts to encoded features"""
        num_nodes = expert_indices.size(0)
        output_dim = self.config.hidden_dim

        # dubug：打印所有专家输入 shape
        for expert_name, expert_input in expert_inputs.items():
            print(f"[Check] expert_inputs['{expert_name}'] shape: {expert_input.shape}")
        
        # Initialize output tensor
        expert_outputs = torch.zeros(num_nodes, output_dim, device=expert_gates.device)

        # Apply each expert
        for i in range(num_nodes):
            for j, (expert_idx, gate) in enumerate(zip(expert_indices[i], expert_gates[i])):
                
                expert_keys = list(self.experts.keys())
                print(f"[Debug] expert_idx: {expert_idx}, expert_keys: {expert_keys}")
                assert expert_idx < len(expert_keys), f"expert_idx {expert_idx} out of range for experts {expert_keys}"
        
                expert_name = list(self.experts.keys())[expert_idx]
                expert = self.experts[expert_name]
                # 取该 expert 对应的输入
                expert_input = expert_inputs[expert_name][i:i+1]
                
                # 打印和断言
                print(f"[Debug] expert_input shape for {expert_name}: {expert_input.shape}")
                assert expert_input.shape[1] == self.config.expert_hidden_dim, \
                    f"{expert_name} expert_input.shape[1] ({expert_input.shape[1]}) != expert_hidden_dim ({self.config.expert_hidden_dim})"
                # 检查专家MLP的输入层
                mlp_in_features = expert.mlp[0].in_features
                print(f"[Debug] {expert_name} expert MLP in_features: {mlp_in_features}")
                if expert_input.shape[1] != mlp_in_features:
                    print(f"[ERROR] {expert_name} expert_input.shape[1]={expert_input.shape[1]}, but mlp_in_features={mlp_in_features}")
                assert mlp_in_features == self.config.expert_hidden_dim, \
                    f"{expert_name} expert MLP in_features ({mlp_in_features}) != expert_hidden_dim ({self.config.expert_hidden_dim})"  
                
                # Prepare expert-specific inputs
                expert_kwargs = {}
                if expert_name == 'action' and 'joint_positions' in batch:
                    expert_kwargs['joint_positions'] = batch['joint_positions'][i:i+1]
                elif expert_name == 'spatial' and 'positions' in batch:
                    expert_kwargs['positions'] = batch['positions'][i:i+1]
                elif expert_name == 'temporal' and 'timestamps' in batch:
                    expert_kwargs['timestamps'] = batch['timestamps'][i:i+1]

                # Apply expert
                expert_output = expert(expert_input, **expert_kwargs)
                
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
        
        print("Batch keys:", batch.keys())
        
        # Encode multimodal inputs
        encoded = self.encode_multimodal_input(batch)

        # Combine encoded features
        combined_features = []
        for modality in ['text', 'tabular', 'structured']:
            if modality in encoded:
                print(f"{modality} to be concatenated shape:", encoded[modality].shape) # Debugging line
                combined_features.append(encoded[modality])

        if combined_features:
            print("All features to be concatenated shapes:", [f.shape for f in combined_features]) # Debugging line
            combined_features = torch.cat(combined_features, dim=-1)
            print("combined_features shape after cat:", combined_features.shape) # Debugging line
        else:
            raise ValueError("No input modalities found in batch")

        # Apply gating mechanism
        gating_output = self.gating(combined_features)
        expert_indices = gating_output['indices']
        expert_gates = gating_output['gates']

        expert_inputs = {
            'action': encoded['tabular'],      # tabular编码 [batch,512]
            'spatial': encoded['structured'],  # structured编码 [batch,512]
            'temporal': encoded['text'],       # text编码 [batch,512]
            'semantic': encoded['text'],       # text编码 [batch,512]
            'safety': encoded['structured']    # structured编码 [batch,512]
        }
        
        print("combined_features shape:", combined_features.shape)  # 应为 [batch, config.expert_hidden_dim * 3]
        print("expert_inputs['action'] shape:", expert_inputs['action'].shape)  # 应为 [batch, config.expert_hidden_dim]
        
        # Apply selected experts
        expert_outputs = self.apply_experts(
            expert_inputs, expert_indices, expert_gates, batch
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