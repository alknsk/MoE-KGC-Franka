#!/usr/bin/env python3
"""调试YAML处理流程"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessors import YAMLProcessor
from data import KGDataLoader
from config import get_config

def debug_yaml_processing():
    # 加载一个YAML文件测试
    yaml_file = './data/train/yamls/task_config_19.yaml'
    
    processor = YAMLProcessor()
    result = processor.process(yaml_file)
    
    print("YAML处理结果:")
    print(f"实体数量: {sum(len(v) for v in result['entities'].values())}")
    print(f"关系数量: {sum(len(v) for v in result['relations'].values())}")
    
    # 详细打印
    print("\n提取的实体:")
    for entity_type, entities in result['entities'].items():
        print(f"  {entity_type}: {len(entities)} 个")
        if entities:
            print(f"    示例: {entities[0]}")
    
    print("\n提取的关系:")
    for rel_type, relations in result['relations'].items():
        print(f"  {rel_type}: {len(relations)} 个")
        if relations:
            print(f"    示例: {relations[0]}")
    
    # 测试完整流程
    print("\n测试完整数据加载流程:")
    config = get_config()
    loader = KGDataLoader(config)
    
    # 只加载一个目录
    all_data = loader.load_data_from_directory('./data/train')
    
    print(f"\n加载的数据:")
    print(f"  PDF数据: {len(all_data['pdf_data'])} 个")
    print(f"  CSV数据: {len(all_data['csv_data'])} 个")
    print(f"  YAML数据: {len(all_data['yaml_data'])} 个")
    
    # 合并实体
    entities = loader.merge_entities(all_data)
    print(f"\n合并后的实体:")
    for entity_type, entity_list in entities.items():
        print(f"  {entity_type}: {len(entity_list)} 个")

if __name__ == '__main__':
    debug_yaml_processing()