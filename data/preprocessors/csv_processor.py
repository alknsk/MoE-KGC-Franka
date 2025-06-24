import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re

class CSVProcessor:
    """Process CSV files containing robot interaction data"""
    
    def __init__(self):
        self.required_columns = {
            'timestamp', 'action', 'joint_positions', 
            'gripper_state', 'object_id', 'success'
        }
        self.franka_joints = ['joint1', 'joint2', 'joint3', 'joint4', 
                             'joint5', 'joint6', 'joint7']
    
    def validate_csv(self, df: pd.DataFrame) -> bool:
        """Validate CSV contains required columns"""
        columns = set(df.columns)
        missing = self.required_columns - columns
        if missing:
            print(f"Missing required columns: {missing}")
            return False
        return True
    
    def parse_joint_positions(self, joint_str: str) -> np.ndarray:
        """Parse joint position string to numpy array"""
        try:
            # Handle different formats: "[1.0, 2.0, ...]" or "1.0,2.0,..."
            joint_str = joint_str.strip('[]')
            positions = [float(x.strip()) for x in joint_str.split(',')]
            return np.array(positions)
        except:
            return np.zeros(7)
    
    def extract_action_entities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract action entities from CSV data"""
        entities = []
        
        for idx, row in df.iterrows():
            entity = {
                'id': f"action_{idx}",
                'type': 'action',
                'name': row['action'],
                'timestamp': row['timestamp'],
                'attributes': {
                    'joint_positions': self.parse_joint_positions(row['joint_positions']),
                    'gripper_state': row['gripper_state'],
                    'success': row['success']
                }
            }
            entities.append(entity)
        
        return entities
    
    def extract_object_entities(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract object entities from CSV data"""
        unique_objects = df['object_id'].unique()
        entities = []
        
        for obj_id in unique_objects:
            if pd.notna(obj_id):
                entity = {
                    'id': str(obj_id),
                    'type': 'object',
                    'name': f"object_{obj_id}",
                    'attributes': {}
                }
                entities.append(entity)
        
        return entities
    
    def extract_temporal_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract temporal relations between actions"""
        relations = []
        df_sorted = df.sort_values('timestamp')
        
        for i in range(len(df_sorted) - 1):
            relation = {
                'head': f"action_{df_sorted.iloc[i].name}",
                'tail': f"action_{df_sorted.iloc[i+1].name}",
                'type': 'follows',
                'attributes': {
                    'time_diff': (df_sorted.iloc[i+1]['timestamp'] - 
                                 df_sorted.iloc[i]['timestamp'])
                }
            }
            relations.append(relation)
        
        return relations
    
    def extract_interaction_relations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract interaction relations between actions and objects"""
        relations = []
        
        for idx, row in df.iterrows():
            if pd.notna(row['object_id']):
                relation = {
                    'head': f"action_{idx}",
                    'tail': str(row['object_id']),
                    'type': 'interacts_with',
                    'attributes': {
                        'action_type': row['action'],
                        'success': row['success']
                    }
                }
                relations.append(relation)
        
        return relations
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics from CSV data"""
        stats = {
            'num_actions': len(df),
            'num_unique_actions': df['action'].nunique(),
            'success_rate': df['success'].mean(),
            'action_distribution': df['action'].value_counts().to_dict(),
            'temporal_span': (df['timestamp'].max() - df['timestamp'].min())
        }
        return stats
    
    def process(self, csv_path: str) -> Dict[str, Any]:
        """Process CSV file and extract knowledge graph elements"""
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Validate
        if not self.validate_csv(df):
            raise ValueError(f"Invalid CSV format in {csv_path}")
        
        # Extract entities
        action_entities = self.extract_action_entities(df)
        object_entities = self.extract_object_entities(df)
        
        # Extract relations
        temporal_relations = self.extract_temporal_relations(df)
        interaction_relations = self.extract_interaction_relations(df)
        
        # Compute statistics
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
            'source': 'csv',
            'path': csv_path,
            'raw_data': df
        }