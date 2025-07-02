#!/usr/bin/env python3
"""简单测试数据加载"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# 直接测试KGDataLoader
from data import KGDataLoader
from config import get_config

config = get_config()
loader = KGDataLoader(config)

# 检查处理器是否正确初始化
print("检查处理器初始化:")
print(f"pdf_processor: {hasattr(loader, 'pdf_processor')}")
print(f"csv_processor: {hasattr(loader, 'csv_processor')}")
print(f"yaml_processor: {hasattr(loader, 'yaml_processor')}")

# 测试加载
print("\n测试数据加载:")
all_data = loader.load_data_from_directory('./data/train')

# 合并实体
print("\n测试合并实体:")
entities = loader.merge_entities(all_data)
for entity_type, entity_list in entities.items():
    print(f"  {entity_type}: {len(entity_list)} 个")