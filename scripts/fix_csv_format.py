#!/usr/bin/env python3
"""修复CSV文件格式问题，将其转化为能够被读取的csv"""
import pandas as pd
import numpy as np
from pathlib import Path
import ast
import re
import random

def fix_joint_positions(jp_str):
    """修复joint_positions格式"""
    if pd.isna(jp_str):
        return None
    
    # 移除换行符
    jp_str = str(jp_str).replace('\n', ' ')
    
    # 移除多余空格
    jp_str = re.sub(r'\s+', ' ', jp_str)
    
    try:
        # 尝试解析为列表
        values = ast.literal_eval(jp_str)
        if len(values) == 7:
            return str(values)
        else:
            print(f"警告：关节数量不对: {len(values)}")
            return None
    except:
        # 如果解析失败，尝试手动提取数字
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', jp_str)
        if len(numbers) == 7:
            values = [float(n) for n in numbers]
            return str(values)
        else:
            print(f"无法解析: {jp_str[:50]}...")
            return None

ACTIONS = ['grasp', 'move', 'place', 'release', 'push', 'pull']
OBJECTS = ['glass_cup', 'plastic_cup', 'wooden_box', 'metal_can', 'book', 'bottle', 'toy_car']

def random_success_column(n):
    """生成模拟success列"""
    mode = random.choice(['all1', 'all0', 'front1_back0'])
    if mode == 'all1':
        return [1] * n
    elif mode == 'all0':
        return [0] * n
    else:
        split = random.randint(1, n-1) if n > 1 else 1
        return [1]*split + [0]*(n-split)

def fix_csv_file(csv_path):
    """修复单个CSV文件"""
    print(f"修复: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        n = len(df)
        # 随机分配动作和物品
        action = random.choice(ACTIONS)
        obj = random.choice(OBJECTS)
        df['action'] = action
        df['object_id'] = obj
        # 随机生成success列
        df['success'] = random_success_column(n)
        # 其它修复逻辑...
        # 修复joint_positions
        df['joint_positions'] = df['joint_positions'].apply(fix_joint_positions)
        df = df.dropna(subset=['joint_positions'])
        # 修复gripper_state
        if df['gripper_state'].dtype == 'object':
            df['gripper_state'] = df['gripper_state'].map({'close': 0, 'closed': 0, 'open': 1, 'opened': 1})
            df['gripper_state'] = df['gripper_state'].fillna(0).astype(int)
        # 保存修复后的文件
        backup_path = csv_path.with_suffix('.bak')
        csv_path.rename(backup_path)
        df.to_csv(csv_path, index=False)
        print(f"  ✅ 修复完成，动作: {action}，物品: {obj}，保留 {len(df)} 行数据")
        return True
    except Exception as e:
        print(f"  ❌ 修复失败: {e}")
        return False

def main():
    data_dir = Path('./raw_data')  # 直接指向 raw_data 文件夹

    # 修复 raw_data 目录下所有 CSV 文件
    csv_files = list(data_dir.glob('**/*.csv'))  # 包含子目录
    print(f"\n修复 raw_data 中的 {len(csv_files)} 个CSV文件...")

    success_count = 0
    for csv_file in csv_files:
        if fix_csv_file(csv_file):
            success_count += 1

    print(f"raw_data: {success_count}/{len(csv_files)} 修复成功")

if __name__ == '__main__':
    main()