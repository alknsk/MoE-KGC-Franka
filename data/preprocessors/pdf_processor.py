import PyPDF2
import pdfplumber
import re
from typing import List, Dict, Tuple, Any
import numpy as np
from transformers import AutoTokenizer

class PDFProcessor:
    """处理PDF文档用于构建知识图谱"""
    
    def __init__(self, tokenizer_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.entity_patterns = {
            'action': r'\b(grasp|move|pick|place|rotate|push|pull)\b',
            'object': r'\b(cube|sphere|cylinder|tool|gripper)\b',
            'spatial': r'\b(above|below|left|right|front|behind)\b',
            'safety': r'\b(collision|force|limit|safe|danger)\b'
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本内容"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Failed to extract text with both methods: {e2}")
        
        return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体 - 返回统一的字典格式"""
        entities = []
        entity_id = 0
    
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    'id': f"pdf_entity_{entity_type}_{entity_id}",
                    'type': entity_type,
                    'name': match.group(),
                    'text': match.group(),
                    'attributes': {
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'source': 'pdf'
                    }
                }
                entities.append(entity)
                entity_id += 1
    
        return entities

    def extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """提取实体之间的关系 - 使用新的实体格式"""
        relations = []
        sentences = text.split('.')
    
        for sent in sentences:
            sent_entities = []
            for entity in entities:
                if entity['text'].lower() in sent.lower():
                    sent_entities.append(entity)
        
            # 在同一个句子中创建实体之间的关系
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    relation = {
                        'head': sent_entities[i]['id'],
                        'tail': sent_entities[j]['id'],
                        'type': 'co_occurrence',
                        'attributes': {
                            'context': sent.strip()[:100]  # 限制上下文长度
                        }
                    }
                    relations.append(relation)
    
        return relations

    def process(self, pdf_path: str) -> Dict[str, Any]:
        """处理PDF文件，提取知识图谱元素"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
    
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', '', text)
    
        # Extract entities and relations - 新格式
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
    
        # Tokenize text（如果需要）
        tokens = None
        if hasattr(self, 'tokenizer'):
            tokens = self.tokenizer(
                text[:512],  # 限制长度
                truncation=True,
                max_length=512,
                padding='max_length',
                return_tensors='pt'
            )
    
        return {
            'text': text[:1000],  # 限制文本长度避免太大
            'entities': entities,  # 统一的列表格式
            'relations': relations,
            'tokens': tokens,
            'source': 'pdf',
            'path': pdf_path
        }