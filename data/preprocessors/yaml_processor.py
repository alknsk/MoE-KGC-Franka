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
        """Validate task definition against schema"""
        for key, expected_type in self.task_schema.items():
            if key not in task:
                print(f"Missing required field: {key}")
                return False
            if not isinstance(task[key], expected_type):
                print(f"Invalid type for {key}: expected {expected_type}, got {type(task[key])}")
                return False
        return True
    
    def extract_task_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract task entities from YAML data"""
        entities = []
        
        if 'tasks' in data:
            for task_id, task_def in enumerate(data['tasks']):
                if self.validate_task_definition(task_def):
                    entity = {
                        'id': f"task_{task_id}",
                        'type': 'task',
                        'name': task_def['name'],
                        'task_type': task_def['type'],
                        'attributes': {
                            'parameters': task_def['parameters'],
                            'constraints': task_def['constraints']
                        }
                    }
                    entities.append(entity)
        
        return entities
    
    def extract_constraint_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract constraint entities from YAML data"""
        entities = []
        constraint_id = 0
        
        # Extract from tasks
        if 'tasks' in data:
            for task in data['tasks']:
                for constraint in task.get('constraints', []):
                    entity = {
                        'id': f"constraint_{constraint_id}",
                        'type': 'constraint',
                        'name': constraint.get('name', f"constraint_{constraint_id}"),
                        'attributes': constraint
                    }
                    entities.append(entity)
                    constraint_id += 1
        
        # Extract global constraints
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
        """Extract safety-related entities from YAML data"""
        entities = []
        
        if 'safety_limits' in data:
            for limit_name, limit_value in data['safety_limits'].items():
                entity = {
                    'id': f"safety_{limit_name}",
                    'type': 'safety_limit',
                    'name': limit_name,
                    'attributes': {
                        'value': limit_value,
                        'unit': self.infer_unit(limit_name)
                    }
                }
                entities.append(entity)
        
        return entities
    
    def extract_hierarchical_relations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract hierarchical relations from YAML structure"""
        relations = []
        
        # Task hierarchy
        if 'task_hierarchy' in data:
            for parent, children in data['task_hierarchy'].items():
                for child in children:
                    relation = {
                        'head': parent,
                        'tail': child,
                        'type': 'subtask_of',
                        'attributes': {}
                    }
                    relations.append(relation)
        
        # Dependency relations
        if 'dependencies' in data:
            for task, deps in data['dependencies'].items():
                for dep in deps:
                    relation = {
                        'head': task,
                        'tail': dep,
                        'type': 'depends_on',
                        'attributes': {}
                    }
                    relations.append(relation)
        
        return relations
    
    def extract_configuration_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration parameters"""
        config = {}
        
        # Robot configuration
        if 'robot_config' in data:
            config['robot'] = data['robot_config']
        
        # Environment configuration
        if 'environment' in data:
            config['environment'] = data['environment']
        
        # Control parameters
        if 'control_params' in data:
            config['control'] = data['control_params']
        
        return config
    
    def infer_unit(self, param_name: str) -> str:
        """Infer unit from parameter name"""
        unit_mapping = {
            'force': 'N',
            'torque': 'Nm',
            'velocity': 'm/s',
            'acceleration': 'm/s²',
            'position': 'm',
            'angle': 'rad'
        }
        
        for key, unit in unit_mapping.items():
            if key in param_name.lower():
                return unit
        return ''
    
    def process(self, yaml_path: str) -> Dict[str, Any]:
        """Process YAML file and extract knowledge graph elements"""
        # Load YAML data
        data = self.load_yaml(yaml_path)
        
        # Extract entities
        task_entities = self.extract_task_entities(data)
        constraint_entities = self.extract_constraint_entities(data)
        safety_entities = self.extract_safety_entities(data)
        
        # Extract relations
        hierarchical_relations = self.extract_hierarchical_relations(data)
        
        # Extract configuration
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
            'source': 'yaml',
            'path': yaml_path,
            'raw_data': data
        }