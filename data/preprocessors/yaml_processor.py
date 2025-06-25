import yaml
import json
from typing import Dict, List, Any, Union
from pathlib import Path

class YAMLProcessor:
    """处理机器人任务与约束相关YAML配置文件的工具类"""

    def __init__(self):
        # 定义任务schema，规定每个任务必须包含的字段及其类型
        self.task_schema = {
            'name': str,           # 任务名称
            'type': str,           # 任务类型
            'parameters': dict,    # 任务参数
            'constraints': list,   # 任务约束
            'safety_limits': dict  # 安全限制
        }

    def load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """加载YAML文件并返回数据字典
        参数:
            yaml_path: YAML文件路径
        返回:
            dict: 解析后的YAML数据
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data

    def validate_task_definition(self, task: Dict[str, Any]) -> bool:
        """校验单个任务定义是否符合schema要求
        参数:
            task: 单个任务的字典
        返回:
            bool: 是否符合schema
        """
        for key, expected_type in self.task_schema.items():
            if key not in task:
                print(f"缺少必要字段: {key}")
                return False
            if not isinstance(task[key], expected_type):
                print(f"字段类型错误: {key} 应为 {expected_type}, 实际为 {type(task[key])}")
                return False
        return True

    def extract_task_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从YAML数据中提取任务实体
        参数:
            data: YAML数据字典
        返回:
            List[Dict]: 任务实体列表
        """
        entities = []

        if 'tasks' in data:
            for task_id, task_def in enumerate(data['tasks']):
                if self.validate_task_definition(task_def):
                    entity = {
                        'id': f"task_{task_id}",          # 任务实体唯一标识
                        'type': 'task',                   # 实体类型
                        'name': task_def['name'],         # 任务名称
                        'task_type': task_def['type'],    # 任务类型
                        'attributes': {                   # 任务属性
                            'parameters': task_def['parameters'],
                            'constraints': task_def['constraints']
                        }
                    }
                    entities.append(entity)

        return entities

    def extract_constraint_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从YAML数据中提取约束实体（包括任务内和全局约束）
        参数:
            data: YAML数据字典
        返回:
            List[Dict]: 约束实体列表
        """
        entities = []
        constraint_id = 0

        # 提取每个任务下的约束
        if 'tasks' in data:
            for task in data['tasks']:
                for constraint in task.get('constraints', []):
                    entity = {
                        'id': f"constraint_{constraint_id}",  # 约束实体唯一标识
                        'type': 'constraint',                 # 实体类型
                        'name': constraint.get('name', f"constraint_{constraint_id}"),  # 约束名称
                        'attributes': constraint              # 约束属性
                    }
                    entities.append(entity)
                    constraint_id += 1

        # 提取全局约束
        if 'global_constraints' in data:
            for constraint in data['global_constraints']:
                entity = {
                    'id': f"constraint_{constraint_id}",
                    'type': 'constraint',
                    'name': constraint.get('name', f"constraint_{constraint_id}"),
                    'attributes': constraint
                }
                entities.append(entity)
                constraint_id += 1

        return entities

    def extract_safety_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从YAML数据中提取安全限制相关实体
        参数:
            data: YAML数据字典
        返回:
            List[Dict]: 安全限制实体列表
        """
        entities = []

        if 'safety_limits' in data:
            for limit_name, limit_value in data['safety_limits'].items():
                entity = {
                    'id': f"safety_{limit_name}",      # 安全限制实体唯一标识
                    'type': 'safety_limit',            # 实体类型
                    'name': limit_name,                # 限制名称
                    'attributes': {
                        'value': limit_value,          # 限制值
                        'unit': self.infer_unit(limit_name)  # 推断单位
                    }
                }
                entities.append(entity)

        return entities

    def extract_hierarchical_relations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从YAML结构中提取层级关系（如任务层级、依赖关系）
        参数:
            data: YAML数据字典
        返回:
            List[Dict]: 层级关系列表
        """
        relations = []

        # 任务层级关系
        if 'task_hierarchy' in data:
            for parent, children in data['task_hierarchy'].items():
                for child in children:
                    relation = {
                        'head': parent,      # 父任务
                        'tail': child,       # 子任务
                        'type': 'subtask_of',# 关系类型：子任务
                        'attributes': {}
                    }
                    relations.append(relation)

        # 任务依赖关系
        if 'dependencies' in data:
            for task, deps in data['dependencies'].items():
                for dep in deps:
                    relation = {
                        'head': task,        # 当前任务
                        'tail': dep,         # 依赖的任务
                        'type': 'depends_on',# 关系类型：依赖
                        'attributes': {}
                    }
                    relations.append(relation)

        return relations

    def extract_configuration_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """提取配置参数（如机器人、环境、控制参数等）
        参数:
            data: YAML数据字典
        返回:
            dict: 配置参数
        """
        config = {}

        # 机器人配置
        if 'robot_config' in data:
            config['robot'] = data['robot_config']

        # 环境配置
        if 'environment' in data:
            config['environment'] = data['environment']

        # 控制参数
        if 'control_params' in data:
            config['control'] = data['control_params']

        return config

    def infer_unit(self, param_name: str) -> str:
        """根据参数名推断物理单位
        参数:
            param_name: 参数名称
        返回:
            str: 单位字符串
        """
        unit_mapping = {
            'force': 'N',           # 力，牛顿
            'torque': 'Nm',         # 力矩，牛·米
            'velocity': 'm/s',      # 速度，米每秒
            'acceleration': 'm/s²', # 加速度，米每二次方秒
            'position': 'm',        # 位置，米
            'angle': 'rad'          # 角度，弧度
        }

        for key, unit in unit_mapping.items():
            if key in param_name.lower():
                return unit
        return ''

    def process(self, yaml_path: str) -> Dict[str, Any]:
        """主处理流程：加载YAML文件，提取实体、关系和配置信息
        参数:
            yaml_path: YAML文件路径
        返回:
            dict: 包含实体、关系、配置等知识图谱元素
        """
        # 加载YAML数据
        data = self.load_yaml(yaml_path)

        # 提取各类实体
        task_entities = self.extract_task_entities(data)
        constraint_entities = self.extract_constraint_entities(data)
        safety_entities = self.extract_safety_entities(data)

        # 提取层级和依赖关系
        hierarchical_relations = self.extract_hierarchical_relations(data)

        # 提取配置参数
        configuration = self.extract_configuration_data(data)

        return {
            'entities': {
                'tasks': task_entities,
                'constraints': constraint_entities,
                'safety': safety_entities
            },
            'relations': {
                'hierarchical': hierarchical_relations
            },
            'configuration': configuration,
            'source': 'yaml',      # 数据来源
            'path': yaml_path,     # 原始文件路径
            'raw_data': data       # 原始数据（字典）
        }