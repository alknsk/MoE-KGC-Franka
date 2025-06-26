import yaml
import json
from typing import Dict, List, Any, Union
from pathlib import Path

class YAMLProcessor:
    """Process YAML configuration files for robot tasks and constraints"""
    
    def __init__(self):
        self.task_schema = {
            'name': str,
            'type': str,
            'parameters': dict,
            'constraints': list,
            'safety_limits': dict
        }
    
    def load_yaml(self, yaml_path: str) -> Dict[str, Any]:
        """Load YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    
    def validate_task_definition(self, task: Dict[str, Any]) -> bool:
        """Validate task definition against schema - 更宽松的验证"""
        # 只检查必需字段
        required_fields = ['name', 'type']
        for field in required_fields:
            if field not in task:
                print(f"Missing required field: {field}")
                return False
        return True
    
    def extract_task_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract task entities from YAML data"""
        entities = []
        
        if 'tasks' in data and isinstance(data['tasks'], list):
            for task_id, task_def in enumerate(data['tasks']):
                if self.validate_task_definition(task_def):
                    entity = {
                        'id': f"task_{task_def['name']}",  # 使用name作为ID
                        'type': 'task',
                        'name': task_def['name'],
                        'task_type': task_def.get('type', 'unknown'),
                        'attributes': {
                            'parameters': task_def.get('parameters', {}),
                            'constraints': task_def.get('constraints', []),
                            'safety_limits': task_def.get('safety_limits', {})
                        }
                    }
                    entities.append(entity)
                    print(f"提取任务实体: {entity['name']}")
        
        return entities
    
    def extract_constraint_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract constraint entities from YAML data"""
        entities = []
        constraint_id = 0
        
        # Extract from tasks
        if 'tasks' in data and isinstance(data['tasks'], list):
            for task in data['tasks']:
                for constraint in task.get('constraints', []):
                    constraint_type = constraint.get('type', 'unknown')
                    entity = {
                        'id': f"constraint_{constraint_type}_{constraint_id}",
                        'type': 'constraint',
                        'name': constraint_type,
                        'attributes': constraint
                    }
                    entities.append(entity)
                    constraint_id += 1
                    print(f"提取约束实体: {entity['name']}")
        
        # Extract global constraints
        if 'global_constraints' in data:
            for constraint in data['global_constraints']:
                entity = {
                    'id': f"constraint_{constraint.get('name', constraint_id)}",
                    'type': 'constraint',
                    'name': constraint.get('name', f"constraint_{constraint_id}"),
                    'attributes': constraint
                }
                entities.append(entity)
                constraint_id += 1
        
        return entities
    
    def extract_safety_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract safety-related entities from YAML data"""
        entities = []
        
        # 从任务中提取safety_limits
        if 'tasks' in data and isinstance(data['tasks'], list):
            for task in data['tasks']:
                if 'safety_limits' in task:
                    for limit_name, limit_value in task['safety_limits'].items():
                        entity = {
                            'id': f"safety_{limit_name}_{task['name']}",
                            'type': 'safety',
                            'name': f"{limit_name}",
                            'attributes': {
                                'value': limit_value,
                                'unit': self.infer_unit(limit_name),
                                'task': task['name']
                            }
                        }
                        entities.append(entity)
                        print(f"提取安全实体: {entity['name']}")
        
        # 全局safety_limits
        if 'safety_limits' in data:
            for limit_name, limit_value in data['safety_limits'].items():
                entity = {
                    'id': f"safety_{limit_name}",
                    'type': 'safety',
                    'name': limit_name,
                    'attributes': {
                        'value': limit_value,
                        'unit': self.infer_unit(limit_name)
                    }
                }
                entities.append(entity)
        
        return entities
    
    def extract_relations(self, data: Dict[str, Any], entities: Dict[str, List]) -> List[Dict[str, Any]]:
        """提取实体之间的关系 - 使用正确的实体ID"""
        relations = []
    
        # 创建ID映射表，方便查找
        entity_id_map = {}
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # 使用实体的name作为key，完整ID作为value
                entity_id_map[entity['name']] = entity['id']
    
        # 任务与约束的关系
        if 'tasks' in data and isinstance(data['tasks'], list):
            for task in data['tasks']:
                task_name = task['name']
                task_id = entity_id_map.get(task_name, f"task_{task_name}")
            
                # 任务与约束的关系
                constraint_index = 0
                for constraint in task.get('constraints', []):
                    constraint_type = constraint.get('type', 'unknown')
                    # 使用正确的约束ID格式
                    constraint_id = f"constraint_{constraint_type}_{constraint_index}"
                
                    # 查找实际的约束实体ID
                    for c_entity in entities.get('constraint', []):
                        if c_entity['name'] == constraint_type:
                            constraint_id = c_entity['id']
                            break
                
                    relation = {
                        'head': task_id,
                        'tail': constraint_id,
                        'type': 'has_constraint',
                        'attributes': {}
                    }
                    relations.append(relation)
                    constraint_index += 1
            
                # 任务与安全限制的关系
                if 'safety_limits' in task:
                    for limit_name in task['safety_limits']:
                        # 使用正确的安全实体ID
                        safety_id = f"safety_{limit_name}_{task_name}"
                    
                        # 查找实际的安全实体ID
                        for s_entity in entities.get('safety', []):
                            if s_entity['name'] == limit_name and task_name in s_entity['id']:
                                safety_id = s_entity['id']
                                break
                    
                        relation = {
                            'head': task_id,
                            'tail': safety_id,
                            'type': 'has_safety_limit',
                            'attributes': {}
                        }
                        relations.append(relation)
    
        return relations
    
    def infer_unit(self, param_name: str) -> str:
        """Infer unit from parameter name"""
        unit_mapping = {
            'force': 'N',
            'torque': 'Nm',
            'velocity': 'm/s',
            'acceleration': 'm/s²',
            'position': 'm',
            'angle': 'rad',
            'limit': 'm/s²'  # for acceleration_limit
        }
        
        for key, unit in unit_mapping.items():
            if key in param_name.lower():
                return unit
        return ''
    
    def process(self, yaml_path: str) -> Dict[str, Any]:
        """Process YAML file and extract knowledge graph elements"""
        print(f"\n处理YAML文件: {yaml_path}")
        
        # Load YAML data
        data = self.load_yaml(yaml_path)
        
        # Extract entities
        task_entities = self.extract_task_entities(data)
        constraint_entities = self.extract_constraint_entities(data)
        safety_entities = self.extract_safety_entities(data)
        
        # 统一实体格式
        all_entities = []
        all_entities.extend([{'type': 'task', **e} for e in task_entities])
        all_entities.extend([{'type': 'constraint', **e} for e in constraint_entities])
        all_entities.extend([{'type': 'safety', **e} for e in safety_entities])
        
        # Extract relations
        entities_dict = {
            'task': task_entities,
            'constraint': constraint_entities,
            'safety': safety_entities
        }
        relations = self.extract_relations(data, entities_dict)
        
        print(f"提取到 {len(all_entities)} 个实体, {len(relations)} 个关系")
        
        return {
            'entities': all_entities,  # 统一格式的实体列表
            'relations': relations,
            'source': 'yaml',
            'path': yaml_path,
            'raw_data': data
        }