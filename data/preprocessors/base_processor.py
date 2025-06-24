"""
预处理器基类
定义所有预处理器的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import hashlib
import pickle
import numpy as np

logger = logging.getLogger(__name__)


class Entity:
    """实体类"""

    def __init__(
            self,
            id: str,
            text: str,
            type: str,
            properties: Optional[Dict[str, Any]] = None,
            embedding: Optional[np.ndarray] = None
    ):
        self.id = id
        self.text = text
        self.type = type
        self.properties = properties or {}
        self.embedding = embedding

    def __repr__(self):
        return f"Entity(id={self.id}, text={self.text}, type={self.type})"

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'type': self.type,
            'properties': self.properties
        }


class Relation:
    """关系类"""

    def __init__(
            self,
            id: str,
            head_id: str,
            tail_id: str,
            type: str,
            properties: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.head_id = head_id
        self.tail_id = tail_id
        self.type = type
        self.properties = properties or {}

    def __repr__(self):
        return f"Relation(id={self.id}, type={self.type}, {self.head_id}->{self.tail_id})"

    def to_dict(self):
        return {
            'id': self.id,
            'head_id': self.head_id,
            'tail_id': self.tail_id,
            'type': self.type,
            'properties': self.properties
        }


class KnowledgeGraph:
    """知识图谱类"""

    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_index: Dict[str, List[str]] = {}  # type -> entity_ids
        self.relation_index: Dict[str, List[str]] = {}  # type -> relation_ids

    def add_entity(self, entity: Entity):
        """添加实体"""
        self.entities[entity.id] = entity
        if entity.type not in self.entity_index:
            self.entity_index[entity.type] = []
        self.entity_index[entity.type].append(entity.id)

    def add_relation(self, relation: Relation):
        """添加关系"""
        self.relations.append(relation)
        if relation.type not in self.relation_index:
            self.relation_index[relation.type] = []
        self.relation_index[relation.type].append(relation.id)

    def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """获取实体的邻居"""
        neighbors = []
        for relation in self.relations:
            if relation.head_id == entity_id:
                neighbors.append((relation.tail_id, relation.type, 'out'))
            elif relation.tail_id == entity_id:
                neighbors.append((relation.head_id, relation.type, 'in'))
        return neighbors

    def to_dict(self):
        return {
            'entities': {eid: e.to_dict() for eid, e in self.entities.items()},
            'relations': [r.to_dict() for r in self.relations]
        }


class BaseProcessor(ABC):
    """预处理器基类"""

    def __init__(self, config):
        self.config = config
        self.cache_dir = Path(config.cache_dir) / self.__class__.__name__
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化实体和关系类型
        self.entity_types = config.entity_types
        self.relation_types = config.relation_types

    @abstractmethod
    def process(self, file_path: Path) -> KnowledgeGraph:
        """处理文件，返回知识图谱"""
        pass

    @abstractmethod
    def extract_entities(self, content: Any) -> List[Entity]:
        """从内容中提取实体"""
        pass

    @abstractmethod
    def extract_relations(self, content: Any, entities: List[Entity]) -> List[Relation]:
        """从内容中提取关系"""
        pass

    def _get_cache_key(self, file_path: Path) -> str:
        """生成缓存键"""
        stat = file_path.stat()
        content = f"{file_path.absolute()}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[KnowledgeGraph]:
        """从缓存加载"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    kg = pickle.load(f)
                logger.info(f"Loaded from cache: {cache_key}")
                return kg
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None

    def _save_to_cache(self, cache_key: str, kg: KnowledgeGraph):
        """保存到缓存"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(kg, f)
            logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def process_with_cache(self, file_path: Path) -> KnowledgeGraph:
        """带缓存的处理"""
        cache_key = self._get_cache_key(file_path)

        # 尝试从缓存加载
        kg = self._load_from_cache(cache_key)
        if kg is not None:
            return kg

        # 处理文件
        kg = self.process(file_path)

        # 保存到缓存
        self._save_to_cache(cache_key, kg)

        return kg