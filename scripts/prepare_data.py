#!/usr/bin/env python3
"""数据预处理脚本"""

import argparse
import os
import sys
from pathlib import Path
import shutil
import pandas as pd
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import KGDataLoader
from config import get_config


def organize_data_files(raw_data_dir: str, output_dir: str, split_ratio: tuple = (0.7, 0.15, 0.15)):
    """
    组织原始数据文件到训练/验证/测试集

    Args:
        raw_data_dir: 原始数据目录
        output_dir: 输出目录
        split_ratio: 训练/验证/测试集比例
    """
    raw_path = Path(raw_data_dir)
    out_path = Path(output_dir)

    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        for dtype in ['pdfs', 'csvs', 'yamls']:
            (out_path / split / dtype).mkdir(parents=True, exist_ok=True)

    # 收集所有文件
    pdf_files = list(raw_path.glob('**/*.pdf'))
    csv_files = list(raw_path.glob('**/*.csv'))
    yaml_files = list(raw_path.glob('**/*.yaml')) + list(raw_path.glob('**/*.yml'))

    print(f"找到 {len(pdf_files)} 个PDF文件")
    print(f"找到 {len(csv_files)} 个CSV文件")
    print(f"找到 {len(yaml_files)} 个YAML文件")

    # 计算分割点
    def split_files(files, ratios):
        n = len(files)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        return {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

    # 分割并复制文件
    for file_type, files, subdir in [
        ('PDF', pdf_files, 'pdfs'),
        ('CSV', csv_files, 'csvs'),
        ('YAML', yaml_files, 'yamls')
    ]:
        splits = split_files(files, split_ratio)
        for split_name, split_files in splits.items():
            for file in split_files:
                dest = out_path / split_name / subdir / file.name
                shutil.copy2(file, dest)
                print(f"复制 {file.name} 到 {split_name}/{subdir}")


def validate_csv_format(csv_path: str) -> bool:
    """验证CSV文件格式"""
    required_columns = {
        'timestamp', 'action', 'joint_positions',
        'gripper_state', 'object_id', 'success'
    }

    try:
        df = pd.read_csv(csv_path)
        columns = set(df.columns)
        missing = required_columns - columns

        if missing:
            print(f"警告: {csv_path} 缺少列: {missing}")
            return False
        return True
    except Exception as e:
        print(f"错误: 无法读取 {csv_path}: {e}")
        return False


def validate_yaml_format(yaml_path: str) -> bool:
    """验证YAML文件格式"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # 检查必要字段
        if 'tasks' not in data:
            print(f"警告: {yaml_path} 缺少 'tasks' 字段")
            return False

        return True
    except Exception as e:
        print(f"错误: 无法读取 {yaml_path}: {e}")
        return False


def preprocess_and_cache(data_dir: str, config_path: str):
    """预处理数据并缓存"""
    config = get_config(config_path)
    loader = KGDataLoader(config)

    # 处理每个数据集
    for split in ['train', 'val', 'test']:
        split_dir = Path(data_dir) / split
        if not split_dir.exists():
            continue

        print(f"\n处理 {split} 数据集...")

        # 加载和处理数据
        all_data = loader.load_data_from_directory(str(split_dir))

        # 合并实体和关系
        entities = loader.merge_entities(all_data)
        relations = loader.merge_relations(all_data)

        # 构建知识图谱
        graph = loader.build_knowledge_graph(entities, relations)

        # 保存处理后的数据
        cache_path = split_dir / 'processed_kg.pkl'
        loader.graph = graph
        loader.entities = entities
        loader.relations = relations
        loader.save_processed_data(str(cache_path))

        print(f"{split} 数据集统计:")
        print(f"  - 实体数: {sum(len(v) for v in entities.values())}")
        print(f"  - 关系数: {len(relations)}")
        print(f"  - 图节点数: {graph.number_of_nodes()}")
        print(f"  - 图边数: {graph.number_of_edges()}")

        # 可视化知识图谱样例
        if split == 'train':
            from utils.kg_visualization import KnowledgeGraphVisualizer
            visualizer = KnowledgeGraphVisualizer(config)
            net = visualizer.create_from_model_output(graph, entities, relations)
            visualizer.save_with_legend(net, str(split_dir / 'kg_sample.html'))
            print(f"  - 知识图谱可视化保存至: {split_dir / 'kg_sample.html'}")


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument('--raw_data', type=str, required=True,
                        help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='输出目录')
    parser.add_argument('--split_ratio', type=float, nargs=3,
                        default=[0.7, 0.15, 0.15],
                        help='训练/验证/测试集比例')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--validate', action='store_true',
                        help='验证数据格式')
    parser.add_argument('--preprocess', action='store_true',
                        help='预处理并缓存数据')

    args = parser.parse_args()

    # 组织数据文件
    print("组织数据文件...")
    organize_data_files(args.raw_data, args.output_dir, tuple(args.split_ratio))

    # 验证数据格式
    if args.validate:
        print("\n验证数据格式...")
        data_path = Path(args.output_dir)

        # 验证CSV文件
        for csv_file in data_path.glob('**/*.csv'):
            validate_csv_format(str(csv_file))

        # 验证YAML文件
        for yaml_file in data_path.glob('**/*.yaml'):
            validate_yaml_format(str(yaml_file))

    # 预处理数据
    if args.preprocess:
        print("\n预处理数据...")
        preprocess_and_cache(args.output_dir, args.config)

    print("\n数据准备完成！")


if __name__ == '__main__':
    main()