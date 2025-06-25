import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re

class CSVProcessor:
    """处理包含机器人交互数据的CSV文件的工具类"""
    
    def __init__(self):
        # 定义CSV文件中必须包含的字段
        self.required_columns = {
            'timestamp', 'action', 'joint_positions', 
            'gripper_state', 'object_id', 'success'
        }
        # Franka机械臂的关节名称列表
        self.franka_joints = ['joint1', 'joint2', 'joint3', 'joint4', 
                             'joint5', 'joint6', 'joint7']
    
    def validate_csv(self, df: pd.DataFrame) -> bool:
        """
        校验CSV文件是否包含所有必需的字段
        参数:
            df: 读取的pandas DataFrame
        返回:
            bool: 是否包含所有必需字段
        """
        columns = set(df.columns)
        missing = self.required_columns - columns
        if missing:
            print(f"缺少必要字段: {missing}")
            return False
        return True
    
    def parse_joint_positions(self, joint_str: str) -> np.ndarray:
        """
        将关节位置的字符串解析为numpy数组
        支持格式: "[1.0, 2.0, ...]" 或 "1.0,2.0,..."
        参数:
            joint_str: 关节位置字符串
        返回:
            np.ndarray: 关节位置数组
        """
        try:
            # 去除方括号并按逗号分割
            joint_str = joint_str.strip('[]')
            positions = [float(x.strip()) for x in joint_str.split(',')]
            return np.array(positions)
        except:
            # 解析失败时返回全零数组
            return np.zeros(7)
    
    def extract_action_entities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        从CSV数据中提取动作实体
        每一行数据对应一个动作实体
        参数:
            df: pandas DataFrame
        返回:
            List[Dict]: 动作实体列表
        """
        entities = []
        
        for idx, row in df.iterrows():
            entity = {
                'id': f"action_{idx}",  # 动作实体唯一标识
                'type': 'action',       # 实体类型
                'name': row['action'],  # 动作名称
                'timestamp': row['timestamp'],  # 时间戳
                'attributes': {         # 其他属性
                    'joint_positions': self.parse_joint_positions(row['joint_positions']),
                    'gripper_state': row['gripper_state'],
                    'success': row['success']
                }
            }
            entities.append(entity)
        
        return entities
    
    def extract_object_entities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        从CSV数据中提取对象实体
        每个唯一object_id对应一个对象实体
        参数:
            df: pandas DataFrame
        返回:
            List[Dict]: 对象实体列表
        """
        unique_objects = df['object_id'].unique()
        entities = []
        
        for obj_id in unique_objects:
            if pd.notna(obj_id):
                entity = {
                    'id': str(obj_id),           # 对象实体唯一标识
                    'type': 'object',            # 实体类型
                    'name': f"object_{obj_id}",  # 对象名称
                    'attributes': {}             # 可扩展属性
                }
                entities.append(entity)
        
        return entities
    
    def extract_temporal_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        提取动作之间的时序关系（如先后顺序）
        参数:
            df: pandas DataFrame
        返回:
            List[Dict]: 时序关系列表
        """
        relations = []
        # 按时间戳排序
        df_sorted = df.sort_values('timestamp')
        
        for i in range(len(df_sorted) - 1):
            relation = {
                'head': f"action_{df_sorted.iloc[i].name}",      # 前一个动作
                'tail': f"action_{df_sorted.iloc[i+1].name}",    # 后一个动作
                'type': 'follows',                               # 关系类型：跟随
                'attributes': {
                    'time_diff': (df_sorted.iloc[i+1]['timestamp'] - 
                                 df_sorted.iloc[i]['timestamp']) # 时间差
                }
            }
            relations.append(relation)
        
        return relations
    
    def extract_interaction_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        提取动作与对象之间的交互关系
        参数:
            df: pandas DataFrame
        返回:
            List[Dict]: 交互关系列表
        """
        relations = []
        
        for idx, row in df.iterrows():
            if pd.notna(row['object_id']):
                relation = {
                    'head': f"action_{idx}",         # 动作实体
                    'tail': str(row['object_id']),   # 对象实体
                    'type': 'interacts_with',        # 关系类型：交互
                    'attributes': {
                        'action_type': row['action'],
                        'success': row['success']
                    }
                }
                relations.append(relation)
        
        return relations
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        统计CSV数据的基本信息
        包括动作数量、唯一动作数、成功率、动作分布、时间跨度等
        参数:
            df: pandas DataFrame
        返回:
            Dict: 统计信息
        """
        stats = {
            'num_actions': len(df),  # 总动作数
            'num_unique_actions': df['action'].nunique(),  # 不同动作种类数
            'success_rate': df['success'].mean(),          # 成功率
            'action_distribution': df['action'].value_counts().to_dict(),  # 各动作出现次数
            'temporal_span': (df['timestamp'].max() - df['timestamp'].min()) # 时间跨度
        }
        return stats
    
    def process(self, csv_path: str) -> Dict[str, Any]:
        """
        主处理流程：读取CSV文件，校验格式，提取实体与关系，统计信息
        参数:
            csv_path: CSV文件路径
        返回:
            Dict: 包含实体、关系、统计信息等的知识图谱元素
        """
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 校验字段
        if not self.validate_csv(df):
            raise ValueError(f"Invalid CSV format in {csv_path}")
        
        # 提取动作和对象实体
        action_entities = self.extract_action_entities(df)
        object_entities = self.extract_object_entities(df)
        
        # 提取时序关系和交互关系
        temporal_relations = self.extract_temporal_relations(df)
        interaction_relations = self.extract_interaction_relations(df)
        
        # 统计信息
        statistics = self.compute_statistics(df)
        
        return {
            'entities': {
                'actions': action_entities,
                'objects': object_entities
            },
            'relations': {
                'temporal': temporal_relations,
                'interactions': interaction_relations
            },
            'statistics': statistics,
            'source': 'csv',      # 数据来源
            'path': csv_path,     # 原始文件路径
            'raw_data': df        # 原始数据（DataFrame）
        }