import time
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
from utils.logger import get_logger

class MoEKGC(nn.Module):
    """
    Mixture of Experts model for Knowledge Graph Construction
    Specialized for Franka robot human-robot interaction
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = get_logger('moe_kgc')
        self.fallback_projection = None

        # 动态设备分配
        self.devices = self._get_available_devices(6)  # 需要6个设备

        self.logger.info(f"成功分配设备：{[str(d) for d in self.devices]}")

        self._monitor_gpu_memory()

        # 使用分配的设备
        encoder_devices = self.devices[:6]
        expert_devices = self.devices[1:6] if len(self.devices) >= 6 else self.devices[1:] * 2
        
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

        expert_devices = self.devices[1:6] if len(self.devices) >= 6 else [self.devices[0]] * 5
        
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

        if hasattr(config, 'memory_optimization') and config.memory_optimization.gradient_checkpointing:
            self.use_checkpointing = True
            # 对适合检查点的模块启用
            if hasattr(self.text_encoder, 'transformer'):
                self.text_encoder.transformer.gradient_checkpointing_enable()
        else:
            self.use_checkpointing = False

    def _monitor_gpu_memory(self):
        """监控GPU内存使用情况"""
        if torch.cuda.is_available():
            for device in self.devices:
                if device.type == 'cuda':
                    gpu_id = device.index
                    try:
                        allocated = torch.cuda.memory_allocated(gpu_id)
                        cached = torch.cuda.memory_reserved(gpu_id)
                        total = torch.cuda.get_device_properties(gpu_id).total_memory
                        self.logger.info(f"GPU {gpu_id} 内存状态: "
                                    f"已分配={allocated/(1024**3):.2f}GB, "
                                    f"已缓存={cached/(1024**3):.2f}GB, "
                                    f"总计={total/(1024**3):.2f}GB")
                    except Exception as e:
                        self.logger.warning(f"无法监控GPU {gpu_id}: {e}")

    def encode_multimodal_input(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode different modalities of input data"""

        # 清理GPU缓存以避免内存不足
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        encoded = {}
        num_nodes = batch['node_features'].shape[0]
        assert batch['text_inputs']['input_ids'].shape[0] == num_nodes
        assert batch['tabular_inputs']['numerical'].shape[0] == num_nodes
        assert batch['structured_inputs']['task_features'].shape[0] == num_nodes

        # Encode text data if available
        if 'text_inputs' in batch:
            self.logger.debug(f"input_ids shape:{batch['text_inputs']['input_ids'].shape}")
            self.logger.debug(f"attention_mask shape:{batch['text_inputs']['attention_mask'].shape}")
            # 检查是否为二维
            assert batch['text_inputs']['input_ids'].dim() == 2, \
                f"input_ids shape must be [num_nodes, seq_len], got {batch['text_inputs']['input_ids'].shape}"
            assert batch['text_inputs']['attention_mask'].dim() == 2, \
                f"attention_mask shape must be [num_nodes, seq_len], got {batch['text_inputs']['attention_mask'].shape}"
            text_input_ids = batch['text_inputs']['input_ids'].to(next(self.text_encoder.parameters()).device)
            text_attention_mask = batch['text_inputs']['attention_mask'].to(next(self.text_encoder.parameters()).device)
            text_encoded = self.text_encoder(text_input_ids, text_attention_mask)
            self.logger.debug(f"text_encoded shape:{text_encoded.shape}")  # Debugging line
            encoded['text'] = text_encoded.to(next(self.gating.parameters()).device)  # gating下一步用

        # Encode tabular data if available
        if 'tabular_inputs' in batch:
            tabular_num = batch['tabular_inputs']['numerical'].to(next(self.tabular_encoder.parameters()).device)
            tabular_cat = batch['tabular_inputs']['categorical']  # 如有必要递归to
            
            tabular_cat_device = {}
            for k, v in tabular_cat.items():
                tabular_cat_device[k] = v.to(next(self.tabular_encoder.parameters()).device)
        
            tabular_encoded = self.tabular_encoder(tabular_num, tabular_cat_device)
            encoded['tabular'] = tabular_encoded.to(next(self.gating.parameters()).device)
            self.logger.debug(f"tabular_encoded shape:{tabular_encoded.shape}") # Debugging line

        elif 'node_features' in batch:
            # 只有没有tabular_inputs时才用node_features
            self.logger.debug(f"node_features shape:{batch['node_features'].shape}")
            # 这里建议只在特殊情况用，并确保shape为(batch, D)
            encoded['tabular'] = batch['node_features'].to(next(self.gating.parameters()).device)
        
        # Encode structured data if available
        if 'structured_inputs' in batch:
            struct_task = batch['structured_inputs']['task_features'].to(next(self.structured_encoder.parameters()).device)
            struct_constraint = batch['structured_inputs']['constraint_features'].to(next(self.structured_encoder.parameters()).device)
            struct_safety = batch['structured_inputs']['safety_features'].to(next(self.structured_encoder.parameters()).device)
            structured_encoded = self.structured_encoder(struct_task, struct_constraint, struct_safety)
            
            self.logger.debug(f"structured_encoded shape:{structured_encoded.shape}") # Debugging line
            
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
        output_dim = self.config.hidden_dim

        # dubug：打印所有专家输入 shape
        for expert_name, expert_input in expert_inputs.items():
            self.logger.debug(f"expert_inputs['{expert_name}'] shape: {expert_input.shape}")
        
        # Initialize output tensor
        expert_outputs = torch.zeros(num_nodes, output_dim, device=expert_gates.device)
        
        # Apply each expert
        for i in range(num_nodes):
            for j, (expert_idx, gate) in enumerate(zip(expert_indices[i], expert_gates[i])):
                
                expert_keys = list(self.experts.keys())
                self.logger.debug(f"[Debug] expert_idx: {expert_idx}, expert_keys: {expert_keys}")
                assert expert_idx < len(expert_keys), f"expert_idx {expert_idx} out of range for experts {expert_keys}"
        
                expert_name = list(self.experts.keys())[expert_idx]
                expert = self.experts[expert_name]
                # 取该 expert 对应的输入
                expert_input = expert_inputs[expert_name][i:i+1]
                
                # 打印和断言
                self.logger.debug(f"[Debug] expert_input shape for {expert_name}: {expert_input.shape}")
                assert expert_input.shape[1] == self.config.expert_hidden_dim, \
                    f"{expert_name} expert_input.shape[1] ({expert_input.shape[1]}) != expert_hidden_dim ({self.config.expert_hidden_dim})"
                # 检查专家MLP的输入层
                mlp_in_features = expert.mlp[0].in_features
                self.logger.debug(f"[Debug] {expert_name} expert MLP in_features: {mlp_in_features}")
                if expert_input.shape[1] != mlp_in_features:
                    self.logger.debug(f"[ERROR] {expert_name} expert_input.shape[1]={expert_input.shape[1]}, but mlp_in_features={mlp_in_features}")
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
        # 开始时清理所有GPU缓存
        if torch.cuda.is_available():
            for device in self.devices:
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        start_time = time.time()

        # 统一处理PyG Batch对象
        if isinstance(batch, Data) or hasattr(batch, 'edge_index'):
            batch_dict = self._convert_pyg_batch_to_dict(batch, task)
        else:
            batch_dict = batch

        convert_time = time.time()
        
        # 检查必要的键
        required_keys = ['node_features', 'text_inputs', 'tabular_inputs', 'structured_inputs']
        for key in required_keys:
            if key not in batch_dict:
                self.logger.warning(f"Missing key in batch: {key}")
        
        # 编码多模态输入
        try:
            encoded = self.encode_multimodal_input(batch_dict)
        except Exception as e:
            self.logger.error(f"多模态编码错误: {str(e)}")
            # 强制清理所有GPU缓存
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass

            # 提供一个合理的回退方案
            if 'node_features' in batch_dict:
                self.logger.warning("使用node_features作为回退")
                node_features = batch_dict['node_features'].to(next(self.gating.parameters()).device)
                # 确保维度匹配门控网络的期望输入
                expected_dim = self.config.expert_hidden_dim  # 128
                current_dim = node_features.size(-1)  # 512
                
                if current_dim != expected_dim:
                    # 使用线性投影将node_features投影到期望维度
                    if not hasattr(self, 'fallback_projection'):
                        self.fallback_projection = nn.Linear(current_dim, expected_dim).to(node_features.device)
                    
                    projected_features = self.fallback_projection(node_features)
                    self.logger.warning(f"将node_features从{current_dim}维投影到{expected_dim}维")
                else:
                    projected_features = node_features
                
                # 创建三个相同的编码来模拟多模态输入
                encoded = {
                    'text': projected_features,
                    'tabular': projected_features, 
                    'structured': projected_features
                }
            else:
                raise RuntimeError(f"无法处理输入数据: {str(e)}")
        
        encode_time = time.time()

        # 合并编码特征
        combined_features = self._combine_encoded_features(encoded)
        
        # 应用门控机制
        gating_output = self.gating(combined_features)
        expert_indices = gating_output['indices']
        expert_gates = gating_output['gates']
        # gating_output['gates'] 在 gating.device 上，后续要搬到 experts/gnn 所在卡
        
        gating_time = time.time()

        # 清理不再需要的张量以节省内存
        del combined_features
        if hasattr(self, 'use_checkpointing') and self.use_checkpointing:
            torch.cuda.empty_cache()  # 在检查点模式下清理缓存
        
        # 准备专家输入
        expert_inputs = self._prepare_expert_inputs(encoded)
        
        # 应用专家（批处理版本）
        expert_outputs = self.apply_experts_batch(
            expert_inputs, expert_indices, expert_gates, batch_dict
        )
        
        expert_time = time.time()

        # 清理不再需要的张量
        del expert_inputs, expert_indices, expert_gates
        if hasattr(self, 'use_checkpointing') and self.use_checkpointing:
            torch.cuda.empty_cache()

        # 应用GNN（支持批处理）
        if 'edge_index' in batch_dict:
            gnn_output = self._apply_gnn_batch(expert_outputs, batch_dict)
            node_embeddings = gnn_output['node_embeddings']
            graph_embedding = gnn_output['graph_embedding']
        else:
            node_embeddings = expert_outputs.to(next(self.link_prediction_head.parameters()).device)
            graph_embedding =  expert_outputs.mean(dim=0, keepdim=True).to(next(self.link_prediction_head.parameters()).device)
        
        gnn_time = time.time()

        # 任务特定的输出头
        output = self._apply_task_head(node_embeddings, batch_dict, task)
        
        head_time = time.time()

        # 添加辅助输出
        output['gating_loss'] = gating_output['load_balancing_loss']
        output['expert_utilization'] = gating_output.get('all_scores', None)

        # 记录性能分析
        if hasattr(self, 'profile') and self.profile:
            self.logger.info(f"性能分析 (批次大小={batch_dict['node_features'].size(0)}):")
            self.logger.info(f"  批次转换: {(convert_time-start_time)*1000:.2f}ms")
            self.logger.info(f"  多模态编码: {(encode_time-convert_time)*1000:.2f}ms")
            self.logger.info(f"  门控机制: {(gating_time-encode_time)*1000:.2f}ms")
            self.logger.info(f"  专家应用: {(expert_time-gating_time)*1000:.2f}ms")
            self.logger.info(f"  GNN处理: {(gnn_time-expert_time)*1000:.2f}ms")
            self.logger.info(f"  任务头: {(head_time-gnn_time)*1000:.2f}ms")
            self.logger.info(f"  总计: {(head_time-start_time)*1000:.2f}ms")
        
        return output

    def _convert_pyg_batch_to_dict(self, batch: Data, task: str) -> Dict[str, Any]:
        """将PyG Batch对象转换为字典格式"""
        batch_dict = {
            'node_features': batch.x if hasattr(batch, 'x') else batch.node_features,
            'edge_index': batch.edge_index,
            'batch_idx': batch.batch if hasattr(batch, 'batch') else torch.zeros(batch.num_nodes, dtype=torch.long)
        }
        # Remap edge_index if n_id exists (NeighborLoader/LinkNeighborLoader)
        if hasattr(batch, 'n_id'):
            n_id = batch.n_id
            n_id_map = {int(n.item()): i for i, n in enumerate(n_id)}
            edge_index = batch.edge_index.clone()
            for i in range(edge_index.shape[1]):
                edge_index[0, i] = n_id_map.get(int(edge_index[0, i].item()), -1)
                edge_index[1, i] = n_id_map.get(int(edge_index[1, i].item()), -1)
            # 过滤掉无效边
            valid_mask = (edge_index[0] >= 0) & (edge_index[1] >= 0)
            edge_index = edge_index[:, valid_mask]
            batch_dict['edge_index'] = edge_index
        
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

                # 添加 relation_ids 处理
                if hasattr(batch, 'relation_ids'):
                    batch_dict['relation_ids'] = batch.relation_ids
                elif hasattr(batch, 'edge_type'):
                    batch_dict['relation_ids'] = batch.edge_type
                elif hasattr(batch, 'rel'):
                    batch_dict['relation_ids'] = batch.rel
                else:
                    # 如果没有关系ID，创建默认的关系ID（假设所有边都是同一种关系）
                    batch_dict['relation_ids'] = torch.zeros(len(batch.head), dtype=torch.long)
                    self.logger.warning("未找到关系ID，使用默认关系ID 0")

            elif hasattr(batch, 'edge_label_index'):
                # 从edge_label_index提取
                pos_mask = batch.edge_label == 1 if hasattr(batch, 'edge_label') else torch.ones(batch.edge_label_index.size(1), dtype=torch.bool)
                batch_dict['head'] = batch.edge_label_index[0, pos_mask]
                batch_dict['tail'] = batch.edge_label_index[1, pos_mask]
                batch_dict['label'] = torch.ones(pos_mask.sum())

                # 添加 relation_ids 处理
                if hasattr(batch, 'edge_label_relation'):
                    batch_dict['relation_ids'] = batch.edge_label_relation[pos_mask]
                elif hasattr(batch, 'edge_type'):
                    batch_dict['relation_ids'] = batch.edge_type[pos_mask] if len(batch.edge_type) > 1 else torch.zeros(pos_mask.sum(), dtype=torch.long)
                else:
                    # 默认关系ID
                    batch_dict['relation_ids'] = torch.zeros(pos_mask.sum(), dtype=torch.long)
                    self.logger.warning("未找到关系ID，使用默认关系ID 0")
        
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
        """
        批处理版本的专家应用
        
        Args:
            expert_inputs: 每个专家的输入特征字典，格式为 {expert_name: tensor}
            expert_indices: 每个样本选择的专家索引，形状为 [batch_size, top_k]
            expert_gates: 每个样本的专家门控权重，形状为 [batch_size, top_k]
            batch: 包含批处理数据的字典
            
        Returns:
            专家输出的加权组合，形状为 [batch_size, output_dim]
        """
        
        batch_size = expert_indices.size(0)
        output_dim = self.config.hidden_dim
        device = next(self.gnn.parameters()).device
        
        # 初始化输出
        expert_outputs = torch.zeros(batch_size, output_dim, device=device)
        
        # 批处理专家应用
        for expert_idx in range(self.config.num_experts):
            expert_name = list(self.experts.keys())[expert_idx]

            # 找出选择了这个专家的样本（在CPU上进行初步检查以提高效率）
            mask_cpu = (expert_indices.cpu() == expert_idx).any(dim=1)
            if not mask_cpu.any():
                continue
                
            expert = self.experts[expert_name]
            expert_device = next(expert.parameters()).device
            
            # 只有确实有样本选择了该专家时，才转移到相应设备
            mask = mask_cpu.to(device)
            
            # 获取对应的输入
            input_tensor = expert_inputs[expert_name]
            mask_on_input_device = mask.to(input_tensor.device)
            expert_input = input_tensor[mask_on_input_device].to(expert_device)
            
            # 准备专家特定的参数
            expert_kwargs = {}
            if expert_name == 'action' and 'joint_positions' in batch:
                jp_tensor = batch['joint_positions']
                mask_on_jp_device = mask.to(jp_tensor.device)
                expert_kwargs['joint_positions'] = jp_tensor[mask_on_jp_device].to(expert_device)# 搬运kwargs到对应设备
            elif expert_name == 'spatial' and 'positions' in batch:
                pos_tensor = batch['positions']
                mask_on_pos_device = mask.to(pos_tensor.device)
                expert_kwargs['positions'] = pos_tensor[mask_on_pos_device].to(expert_device)# 搬运kwargs到对应设备
            elif expert_name == 'temporal' and 'timestamps' in batch:
                ts_tensor = batch['timestamps']
                mask_on_ts_device = mask.to(ts_tensor.device)
                expert_kwargs['timestamps'] = ts_tensor[mask_on_ts_device].to(expert_device)# 搬运kwargs到对应设备
            
            # 应用专家
            # with torch.cuda.amp.autocast(enabled=False):  # 避免混合精度问题
            expert_output = expert(expert_input, **expert_kwargs)
            
            # 获取对应的门控权重
            expert_gate_weights = expert_gates[mask.to(expert_gates.device)]
            gate_idx = (expert_indices[mask.to(expert_gates.device)] == expert_idx).float()
            weights = (expert_gate_weights * gate_idx).sum(dim=1, keepdim=True)
            
            # 全部搬到device
            weights = weights.to(device)
            expert_output = expert_output.to(device)
        
            # expert_output直接搬到gnn卡中计算，再加权输出
            expert_outputs[mask] += weights * expert_output
        
        return expert_outputs

    def _apply_gnn_batch(self, node_features: torch.Tensor, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """批处理GNN应用"""
        device = next(self.gnn.parameters()).device
        node_features = node_features.to(device)
        edge_index = batch_dict['edge_index'].to(device)
        
        batch_idx = batch_dict.get('batch_idx', None)
        if batch_idx is None:
            batch_idx = torch.zeros(node_features.size(0), dtype=torch.long, device=device)
        else:
            batch_idx = batch_idx.to(device)
            
        edge_attr = batch_dict.get('edge_attr', None)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)
        
        # 验证边索引
        num_nodes = node_features.size(0)
        if edge_index.numel() > 0:
            assert edge_index.max() < num_nodes, f"Edge index out of bounds: max {edge_index.max()} >= {num_nodes}"
            assert edge_index.min() >= 0, f"Edge index negative: min {edge_index.min()}"
        
        # 应用GNN
        gnn_output = self.gnn(
            node_features,
            edge_index,
            edge_attr,
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
                
    def _apply_task_head(self, 
                    node_embeddings: torch.Tensor, 
                    batch_dict: Dict[str, Any], 
                    task: str) -> Dict[str, torch.Tensor]:
        """
        应用任务特定的头部网络
        
        Args:
            node_embeddings: 节点嵌入，形状为 [num_nodes, hidden_dim]
            batch_dict: 包含批处理数据的字典
            task: 任务类型，可选值为 'link_prediction', 'entity_classification', 'relation_extraction'
            
        Returns:
            任务头部的输出字典
            
        Raises:
            ValueError: 如果任务类型未知
        """
        # 统一搬到head所在卡
        device = next(self.link_prediction_head.parameters()).device
        node_embeddings = node_embeddings.to(device)

        if task == 'link_prediction':
            # 检查必要字段
            required_fields = ['head', 'tail']
            missing_fields = [field for field in required_fields if field not in batch_dict]
            
            if missing_fields:
                raise ValueError(f"链接预测任务缺少必要字段: {', '.join(missing_fields)}")

            head_indices = batch_dict['head'].to(device)
            tail_indices = batch_dict['tail'].to(device)

            if 'relation_ids' in batch_dict:
                relation_ids = batch_dict['relation_ids'].to(device)
            else:
                # 如果没有关系ID，创建默认的关系ID
                relation_ids = torch.zeros(len(head_indices), dtype=torch.long, device=device)
                self.logger.warning("未找到关系ID，使用默认关系ID 0")
                
            # 增加索引验证
            num_nodes = node_embeddings.size(0)
            if torch.max(head_indices) >= num_nodes or torch.max(tail_indices) >= num_nodes:
                self.logger.error(f"索引超出范围: head_max={torch.max(head_indices).item()}, "
                                    f"tail_max={torch.max(tail_indices).item()}, num_nodes={num_nodes}")
                # 修复索引
                head_indices = torch.clamp(head_indices, 0, num_nodes-1)
                tail_indices = torch.clamp(tail_indices, 0, num_nodes-1)
                
            head_embeddings = node_embeddings[head_indices]
            tail_embeddings = node_embeddings[tail_indices]
                
            return self.link_prediction_head(head_embeddings, tail_embeddings, relation_ids)

        elif task == 'entity_classification':
            if 'node_idx' in batch_dict:
                node_idx = batch_dict['node_idx'].to(device)
                target_embeddings = node_embeddings[node_idx]
                return self.entity_classification_head(target_embeddings)
            else:
                return self.entity_classification_head(node_embeddings)

        elif task == 'relation_extraction':
            if 'head' in batch_dict and 'tail' in batch_dict:
                head_indices = batch_dict['head'].to(device)
                tail_indices = batch_dict['tail'].to(device)
                
                head_embeddings = node_embeddings[head_indices]
                tail_embeddings = node_embeddings[tail_indices]
                
                return self.relation_extraction_head(head_embeddings, tail_embeddings)
            else:
                raise ValueError("Missing required fields for relation_extraction task")
        
        else:
            raise ValueError(f"Unknown task: {task}")

    def _get_available_devices(self, num_required):
        """动态分配可用的GPU设备"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，使用CPU")
            return [torch.device('cpu')] * num_required
        
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"检测到{num_gpus}个GPU设备")
        
        # 如果你看到这里，我这样做的原因是，我现在使用的服务器的显存有限。当使用CUDA_VISIBLE_DEVICES时，设备ID会被重新映射
        # 例如：CUDA_VISIBLE_DEVICES="1,2,3,4,5" 会将原来的1,2,3,4,5映射为0,1,2,3,4
        available_gpu_ids = list(range(num_gpus))
        
        if len(available_gpu_ids) == 0:
            self.logger.error("除0号GPU外没有其他GPU可用，强制使用CPU")
            return [torch.device('cpu')] * num_required

        # 获取每个GPU的内存使用情况
        gpu_memory_usage = []
        for i in available_gpu_ids:
            try:
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i)
                total = torch.cuda.get_device_properties(i).total_memory
                usage_ratio = allocated / total
                gpu_memory_usage.append((i, usage_ratio, allocated, total))
                self.logger.info(f"GPU {i}: 已使用 {allocated/(1024**3):.2f}GB / {total/(1024**3):.2f}GB ({usage_ratio*100:.1f}%)")
            except Exception as e:
                self.logger.warning(f"无法获取GPU {i}的内存信息: {e}")
                gpu_memory_usage.append((i, 1.0, 0, 1))  # 假设已满，避免使用

         # 按内存使用率排序，优先使用空闲的GPU
        gpu_memory_usage.sort(key=lambda x: x[1])
        
        # 过滤掉使用率超过80%的GPU
        available_gpus = [(gpu_id, usage) for gpu_id, usage, _, _ in gpu_memory_usage if usage < 0.8]
        
        if len(available_gpus) >= num_required:
            # 有足够的空闲GPU
            selected_gpus = [gpu_id for gpu_id, _ in available_gpus[:num_required]]
            devices = [torch.device(f'cuda:{gpu_id}') for gpu_id in selected_gpus]
            self.logger.info(f"选择GPU: {selected_gpus}")
            return devices
        elif len(available_gpus) > 0:
            # 部分GPU可用，循环使用
            selected_gpus = [gpu_id for gpu_id, _ in available_gpus]
            devices = []
            for i in range(num_required):
                gpu_id = selected_gpus[i % len(selected_gpus)]
                devices.append(torch.device(f'cuda:{gpu_id}'))
            self.logger.warning(f"可用GPU不足，循环使用GPU: {selected_gpus}，避开0号GPU")
            return devices
        else:
            # 所有GPU都很忙，但仍然避开0号GPU
            fallback_gpus = available_gpu_ids[:num_required] if len(available_gpu_ids) >= num_required else available_gpu_ids
            devices = []
            for i in range(num_required):
                gpu_id = fallback_gpus[i % len(fallback_gpus)]
                devices.append(torch.device(f'cuda:{gpu_id}'))
            self.logger.warning(f"所有可用GPU都繁忙，强制使用GPU: {fallback_gpus}，避开0号GPU")
            return devices