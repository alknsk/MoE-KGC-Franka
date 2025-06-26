#!/usr/bin/env python3
"""验证数据格式"""
import pandas as pd
import yaml
import sys
from pathlib import Path

def validate_csv(csv_path):
    """详细验证CSV文件"""
    print(f"\n验证CSV: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"  行数: {len(df)}")
        print(f"  列: {list(df.columns)}")
        
        # 检查必需列
        required = ['timestamp', 'action', 'joint_positions', 'gripper_state', 'object_id', 'success']
        missing = set(required) - set(df.columns)
        if missing:
            print(f"  ❌ 缺少列: {missing}")
            return False
        
        # 检查数据类型和格式
        issues = []
        
        # 检查joint_positions格式
        for idx, jp in enumerate(df['joint_positions'].head(5)):
            try:
                if pd.isna(jp):
                    issues.append(f"行{idx}: joint_positions为空")
                    continue
                    
                # 尝试解析
                jp_str = str(jp).strip()
                if not (jp_str.startswith('[') and jp_str.endswith(']')):
                    issues.append(f"行{idx}: joint_positions格式错误: {jp_str[:50]}...")
                else:
                    # 解析列表
                    values = eval(jp_str)
                    if len(values) != 7:
                        issues.append(f"行{idx}: joint_positions应有7个值，实际{len(values)}个")
            except Exception as e:
                issues.append(f"行{idx}: 解析joint_positions失败: {e}")
        
        # 检查其他列
        if df['timestamp'].isnull().any():
            issues.append("timestamp列有空值")
        if df['action'].isnull().any():
            issues.append("action列有空值")
            
        if issues:
            print("  ⚠️  发现问题:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✅ 格式正确")
            
        # 显示示例数据
        print("\n  前3行数据:")
        print(df.head(3).to_string())
        
        return len(issues) == 0
        
    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

def validate_yaml(yaml_path):
    """验证YAML文件"""
    print(f"\n验证YAML: {yaml_path}")
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'tasks' not in data:
            print("  ❌ 缺少'tasks'字段")
            return False
            
        print(f"  ✅ 包含 {len(data['tasks'])} 个任务")
        
        # 显示第一个任务
        if data['tasks']:
            print(f"  第一个任务: {data['tasks'][0].get('name', 'unnamed')}")
            
        return True
        
    except Exception as e:
        print(f"  ❌ 解析失败: {e}")
        return False

if __name__ == '__main__':
    data_dir = Path('./data/train')
    
    # 验证CSV
    csv_files = list((data_dir / 'csvs').glob('*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    valid_csv = 0
    for csv_file in csv_files[:]: 
        if validate_csv(csv_file):
            valid_csv += 1
    
    # 验证YAML
    yaml_files = list((data_dir / 'yamls').glob('*.yaml'))
    print(f"\n找到 {len(yaml_files)} 个YAML文件")
    
    valid_yaml = 0
    for yaml_file in yaml_files[:]: 
        if validate_yaml(yaml_file):
            valid_yaml += 1
    
    print(f"\n总结: {valid_csv}/{min(80, len(csv_files))} CSV有效, {valid_yaml}/{min(17, len(yaml_files))} YAML有效")