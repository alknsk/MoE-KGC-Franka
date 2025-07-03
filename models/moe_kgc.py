import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union
from torch_geometric.data import Data
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

        # 主干各部分进行并行计算
        encoder_devices = [
            torch.device('cuda:0'),  # text_encoder
            torch.device('cuda:1'),  # tabular_encoder
            torch.device('cuda:2'),  # structured_encoder
            torch.device('cuda:3'),  # gating
            torch.device('cuda:4'),  # gnn
            torch.device('cuda:5'),  # heads
        ]
        
        # Initialize encoders
        self.text_encoder = TextEncoder(
            model_name="bert-base-uncased",
            hidden_dim=config.hidden_dim,
            output_dim=config.expert_hidden_dim,
            dropout_rate=config.dropout_rate
        ).to(encoder_devices[0])

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
        ).to(encoder_devices[1])

        self.structured_encoder = StructuredEncoder(
            input_dim=256,  # Placeholder
            hidden_dims=[config.expert_hidden_dim, config.expert_hidden_dim // 2],
            output_dim=config.expert_hidden_dim,
            dropout_rate=config.dropout_rate,
            use_graph_structure=True
        ).to(encoder_devices[2])

        expert_devices = [torch.device(f'cuda:{i}') for i in range(1, 6)]  # 1~5号卡
        
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
            ).to(expert_devices[0]),# 给每个专家分配不同的卡进行计算
            'spatial': SpatialExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['spatial_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['spatial_expert'].use_attention,
                workspace_dim=3
            ).to(expert_devices[1]),
            'temporal': TemporalExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['temporal_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['temporal_expert'].use_attention
            ).to(expert_devices[2]),
            'semantic': SemanticExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['semantic_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['semantic_expert'].use_attention,
                vocab_size=config.data.vocab_size
            ).to(expert_devices[3]),
            'safety': SafetyExpert(
                input_dim=expert_input_dim,
                hidden_dims=config.experts['safety_expert'].hidden_dims,
                output_dim=config.hidden_dim,
                dropout_rate=config.dropout_rate,
                use_attention=config.experts['safety_expert'].use_attention
            ).to(expert_devices[4])
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
        ).to(encoder_devices[3])

        # Initialize GNN
        self.gnn = EnhancedGNN(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.graph.num_layers,
            edge_dim=config.graph.edge_hidden_dim if config.graph.use_edge_features else None,
            dropout=config.dropout_rate
        ).to(encoder_devices[4])

        # Initialize graph fusion
        self.graph_fusion = GraphFusion(
            input_dims=[config.hidden_dim] * config.num_experts,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            fusion_type='attention',
            dropout=config.dropout_rate
        ).to(encoder_devices[4])

        # Initialize task heads
        self.link_prediction_head = LinkPredictionHead(
            entity_dim=config.hidden_dim,
            relation_dim=config.hidden_dim // 2,
            hidden_dim=config.hidden_dim,
            num_relations=config.data.num_relations,
            dropout=config.dropout_rate
        ).to(encoder_devices[5])

        self.entity_classification_head = EntityClassificationHead(
            input_dim=config.hidden_dim,
            num_classes=config.data.num_entity_types,
            hidden_dims=[config.hidden_dim, config.hidden_dim // 2],
            dropout=config.dropout_rate
        ).to(encoder_devices[5])

        self.relation_extraction_head = RelationExtractionHead(
            entity_dim=config.hidden_dim,
            num_relations=config.data.num_relations,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate
        ).to(encoder_devices[5])

    def encode_multimodal_input(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode different modalities of input data"""
        encoded = {}
        num_nodes = batch['node_features'].shape[0]
        assert batch['text_inputs']['input_ids'].shape[0] == num_nodes
        assert batch['tabular_inputs']['numerical'].shape[0] == num_nodes
        assert batch['structured_inputs']['task_features'].shape[0] == num_nodes

        # Encode text data if available
        if 'text_inputs' in batch:
            print("input_ids shape:", batch['text_inputs']['input_ids'].shape)
            print("attention_mask shape:", batch['text_inputs']['attention_mask'].shape)
            # 检查是否为二维
            assert batch['text_inputs']['input_ids'].dim() == 2, \
                f"input_ids shape must be [num_nodes, seq_len], got {batch['text_inputs']['input_ids'].shape}"
            assert batch['text_inputs']['attention_mask'].dim() == 2, \
                f"attention_mask shape must be [num_nodes, seq_len], got {batch['text_inputs']['attention_mask'].shape}"
            text_input_ids = batch['text_inputs']['input_ids'].to(next(self.text_encoder.parameters()).device)
            text_attention_mask = batch['text_inputs']['attention_mask'].to(next(self.text_encoder.parameters()).device)
            text_encoded = self.text_encoder(text_input_ids, text_attention_mask)
            print("text_encoded shape:", text_encoded.shape)  # Debugging line
            encoded['text'] = text_encoded.to(next(self.gating.parameters()).device)  # gating下一步用

        # Encode tabular data if available
        if 'tabular_inputs' in batch:
            tabular_num = batch['tabular_inputs']['numerical'].to(next(self.tabular_encoder.parameters()).device)
            tabular_cat = batch['tabular_inputs']['categorical']  # 如有必要递归to
            tabular_encoded = self.tabular_encoder(tabular_num, tabular_cat)
            encoded['tabular'] = tabular_encoded.to(next(self.gating.parameters()).device)
            print("tabular_encoded shape:", tabular_encoded.shape) # Debugging line

        elif 'node_features' in batch:
            # 只有没有tabular_inputs时才用node_features
            print("node_features shape:", batch['node_features'].shape)
            # 这里建议只在特殊情况用，并确保shape为(batch, D)
            encoded['tabular'] = batch['node_features'].to(next(self.gating.parameters()).device)
        
        # Encode structured data if available
        if 'structured_inputs' in batch:
            struct_task = batch['structured_inputs']['task_features'].to(next(self.structured_encoder.parameters()).device)
            struct_constraint = batch['structured_inputs']['constraint_features'].to(next(self.structured_encoder.parameters()).device)
            struct_safety = batch['structured_inputs']['safety_features'].to(next(self.structured_encoder.parameters()).device)
            structured_encoded = self.structured_encoder(struct_task, struct_constraint, struct_safety)
            
            print("structured_encoded shape:", structured_encoded.shape) # Debugging line
            
            encoded['structured'] = structured_encoded.to(next(self.gating.parameters()).device)
            
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
        num_nodes = expert_indices.size(0)
        output_dim = self.config.hidden_dim

        # dubug：打印所有专家输入 shape
        for expert_name, expert_input in expert_inputs.items():
            print(f"[Check] expert_inputs['{expert_name}'] shape: {expert_input.shape}")
        
        # Initialize output tensor
        expert_outputs = torch.zeros(num_nodes, output_dim, device=expert_gates.device)
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

    def forward(self, batch: Union[Dict[str, Any], Data], task: str = 'link_prediction') -> Dict[str, torch.Tensor]:
        """
        支持批处理的前向传播
        
        Args:
            batch: PyG Batch对象或包含批处理数据的字典
            task: 任务类型
        """
        # 统一处理PyG Batch对象
        if isinstance(batch, Data) or hasattr(batch, 'edge_index'):
            batch_dict = self._convert_pyg_batch_to_dict(batch, task)
        else:
            batch_dict = batch
        
        # 检查必要的键
        required_keys = ['node_features', 'text_inputs', 'tabular_inputs', 'structured_inputs']
        for key in required_keys:
            if key not in batch_dict:
                self.logger.warning(f"Missing key in batch: {key}")
        
        # 编码多模态输入
        encoded = self.encode_multimodal_input(batch_dict)
        
        # 合并编码特征
        combined_features = self._combine_encoded_features(encoded)
        
        # 应用门控机制
        gating_output = self.gating(combined_features)
        expert_indices = gating_output['indices']
        expert_gates = gating_output['gates']
        # gating_output['gates'] 在 gating.device 上，后续要搬到 experts/gnn 所在卡
        
        # 准备专家输入
        expert_inputs = self._prepare_expert_inputs(encoded)
        
        # 应用专家（批处理版本）
        expert_outputs = self.apply_experts_batch(
            expert_inputs, expert_indices, expert_gates, batch_dict
        )
        
        # 应用GNN（支持批处理）
        if 'edge_index' in batch_dict:
            gnn_output = self._apply_gnn_batch(expert_outputs, batch_dict)
            node_embeddings = gnn_output['node_embeddings']
            graph_embedding = gnn_output['graph_embedding']
        else:
            node_embeddings = expert_outputs.to(next(self.link_prediction_head.parameters()).device)
            graph_embedding =  expert_outputs.mean(dim=0, keepdim=True).to(next(self.link_prediction_head.parameters()).device)
        
        # 任务特定的输出头
        output = self._apply_task_head(node_embeddings, batch_dict, task)
        
        # 添加辅助输出
        output['gating_loss'] = gating_output['load_balancing_loss']
        output['expert_utilization'] = gating_output.get('all_scores', None)
        
        return output

    def _convert_pyg_batch_to_dict(self, batch: Data, task: str) -> Dict[str, Any]:
        """将PyG Batch对象转换为字典格式"""
        batch_dict = {
            'node_features': batch.x if hasattr(batch, 'x') else batch.node_features,
            'edge_index': batch.edge_index,
            'batch_idx': batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.num_nodes, dtype=torch.long)
        }
        
        # 多模态输入
        for key in ['text_inputs', 'tabular_inputs', 'structured_inputs']:
            if hasattr(batch, key):
                batch_dict[key] = getattr(batch, key)
        
        # 任务特定数据
        if task == 'link_prediction':
            if hasattr(batch, 'head') and hasattr(batch, 'tail'):
                batch_dict['head'] = batch.head
                batch_dict['tail'] = batch.tail
                batch_dict['label'] = batch.label if hasattr(batch, 'label') else torch.ones(len(batch.head))
            elif hasattr(batch, 'edge_label_index'):
                # 从edge_label_index提取
                pos_mask = batch.edge_label == 1 if hasattr(batch, 'edge_label') else torch.ones(batch.edge_label_index.size(1), dtype=torch.bool)
                batch_dict['head'] = batch.edge_label_index[0, pos_mask]
                batch_dict['tail'] = batch.edge_label_index[1, pos_mask]
                batch_dict['label'] = torch.ones(pos_mask.sum())
        
        elif task == 'entity_classification':
            if hasattr(batch, 'node_idx'):
                batch_dict['node_idx'] = batch.node_idx
            else:
                batch_dict['node_idx'] = torch.arange(batch.num_nodes)
            
            if hasattr(batch, 'y'):
                batch_dict['label'] = batch.y
            elif hasattr(batch, 'label'):
                batch_dict['label'] = batch.label
        
        return batch_dict

    def apply_experts_batch(self, expert_inputs: Dict[str, torch.Tensor],
                       expert_indices: torch.Tensor,
                       expert_gates: torch.Tensor,
                       batch: Dict[str, Any]) -> torch.Tensor:
        """批处理版本的专家应用"""
        batch_size = expert_indices.size(0)
        output_dim = self.config.hidden_dim
        device = self.gnn.device
        
        # 初始化输出
        expert_outputs = torch.zeros(batch_size, output_dim, device=device)
        
        # 批处理专家应用
        for expert_idx in range(self.config.num_experts):
            expert_name = list(self.experts.keys())[expert_idx]
            expert = self.experts[expert_name]
            expert_device = next(expert.parameters()).device
            
            # 找出选择了这个专家的样本
            mask = (expert_indices == expert_idx).any(dim=1)
            if not mask.any():
                continue
            
            # 获取对应的输入
            expert_input = expert_inputs[expert_name][mask].to(expert_device) # 搬运输入到expert所在设备
            
            # 准备专家特定的参数
            expert_kwargs = {}
            if expert_name == 'action' and 'joint_positions' in batch:
                expert_kwargs['joint_positions'] = batch['joint_positions'][mask].to(expert_device)# 搬运kwargs到对应设备
            elif expert_name == 'spatial' and 'positions' in batch:
                expert_kwargs['positions'] = batch['positions'][mask].to(expert_device)# 搬运kwargs到对应设备
            elif expert_name == 'temporal' and 'timestamps' in batch:
                expert_kwargs['timestamps'] = batch['timestamps'][mask].to(expert_device)# 搬运kwargs到对应设备
            
            # 应用专家
            # with torch.cuda.amp.autocast(enabled=False):  # 避免混合精度问题
            expert_output = expert(expert_input, **expert_kwargs)
            
            # 获取对应的门控权重
            expert_gate_weights = expert_gates[mask]
            gate_idx = (expert_indices[mask] == expert_idx).float()
            weights = (expert_gate_weights * gate_idx).sum(dim=1, keepdim=True)
            
        # expert_output直接搬到gnn卡中计算，再加权输出
        expert_outputs[mask] += weights * expert_output.to(device)
        
        return expert_outputs

    def _apply_gnn_batch(self, node_features: torch.Tensor, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """批处理GNN应用"""
        node_features = node_features.to(next(self.gnn.parameters()).device)
        edge_index = batch_dict['edge_index']
        batch_idx = batch_dict.get('batch_idx', torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device))
        
        # 验证边索引
        num_nodes = node_features.size(0)
        if edge_index.size(0) > 0:
            assert edge_index.max() < num_nodes, f"Edge index out of bounds: max {edge_index.max()} >= {num_nodes}"
            assert edge_index.min() >= 0, f"Edge index negative: min {edge_index.min()}"
        
        # 应用GNN
        gnn_output = self.gnn(
            node_features,
            edge_index,
            batch_dict.get('edge_attr', None),
            batch_idx
        )
        
        return {
            'node_embeddings': gnn_output['node_embeddings'].to(next(self.link_prediction_head.parameters()).device),
            'graph_embedding': gnn_output['graph_embedding'].to(next(self.link_prediction_head.parameters()).device)
        }

    def _combine_encoded_features(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """合并编码特征"""
        features = []
        for modality in ['text', 'tabular', 'structured']:
            if modality in encoded and encoded[modality] is not None:
                features.append(encoded[modality])
        
        if not features:
            raise ValueError("No encoded features found")
        
        # 确保所有特征有相同的batch大小
        batch_size = features[0].size(0)
        for f in features:
            assert f.size(0) == batch_size, f"Inconsistent batch size: {f.size(0)} vs {batch_size}"
        
        combined = torch.cat(features, dim=-1)
        
        return combined.to(next(self.gating.parameters()).device)

    def _prepare_expert_inputs(self, encoded: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """准备专家输入，确保每个专家都有输入"""
        # 默认映射
        expert_inputs = {
            'action': encoded.get('tabular', encoded.get('text')),
            'spatial': encoded.get('structured', encoded.get('tabular')),
            'temporal': encoded.get('text', encoded.get('structured')),
            'semantic': encoded.get('text', encoded.get('tabular')),
            'safety': encoded.get('structured', encoded.get('tabular'))
        }
        
        # 确保没有None值
        default_input = list(encoded.values())[0]
        for key in expert_inputs:
            if expert_inputs[key] is None:
                expert_inputs[key] = default_input
        
        return expert_inputs

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
                
    def _apply_task_head(self, node_embeddings, batch_dict, task):
        # 统一搬到head所在卡
        node_embeddings = node_embeddings.to(next(self.link_prediction_head.parameters()).device)
        if task == 'link_prediction':
            return self.link_prediction_head(node_embeddings, ...)
        elif task == 'entity_classification':
            return self.entity_classification_head(node_embeddings, ...)
        elif task == 'relation_extraction':
            return self.relation_extraction_head(node_embeddings, ...)
        else:
            raise ValueError(f"Unknown task: {task}")