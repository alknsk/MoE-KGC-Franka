#!/usr/bin/env python3
"""预处理数据为PyG格式"""

import argparse
import torch
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import KGDataLoader
from config import get_config

def main():
    parser = argparse.ArgumentParser(description='预处理数据为PyG格式')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = get_config(args.config)
    
    # 创建数据加载器
    data_loader = KGDataLoader(config)
    
    # 处理每个数据集
    for split in ['train', 'val', 'test']:
        split_dir = Path(args.data_dir) / split
        if not split_dir.exists():
            print(f"跳过 {split} (目录不存在)")
            continue
            
        print(f"\n处理 {split} 数据集...")
        
        # 创建数据集
        dataset = data_loader.create_dataset(split_dir)
        
        # 转换为PyG格式
        pyg_data = dataset.to_pyg_data()
        
        # 保存
        output_path = split_dir / 'pyg_data.pt'
        torch.save(pyg_data, output_path)
        
        print(f"PyG数据保存至: {output_path}")
        print(f"  节点数: {pyg_data.num_nodes}")
        print(f"  边数: {pyg_data.edge_index.size(1)}")
        print(f"  特征维度: {pyg_data.x.shape}")
        
        # 验证多模态数据
        if hasattr(pyg_data, 'text_inputs'):
            print(f"  文本输入: {pyg_data.text_inputs['input_ids'].shape}")
        if hasattr(pyg_data, 'tabular_inputs'):
            print(f"  表格输入: {pyg_data.tabular_inputs['numerical'].shape}")
        if hasattr(pyg_data, 'structured_inputs'):
            print(f"  结构化输入: {pyg_data.structured_inputs['task_features'].shape}")

if __name__ == '__main__':
    main()