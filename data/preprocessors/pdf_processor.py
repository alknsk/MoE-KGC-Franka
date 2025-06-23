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
    
    def extract_entities(self, text: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """从具有位置的文本中提取实体"""
        entities = {entity_type: [] for entity_type in self.entity_patterns}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities[entity_type].append((
                    match.group(),
                    match.start(),
                    match.end()
                ))
        
        return entities
    
    def extract_relations(self, text: str, entities: Dict[str, List[Tuple[str, int, int]]]) -> List[Dict[str, Any]]:
        """提取实体之间的关系"""
        relations = []
        sentences = text.split('.')
        
        for sent in sentences:
            sent_entities = []
            for entity_type, entity_list in entities.items():
                for entity, start, end in entity_list:
                    if entity.lower() in sent.lower():
                        sent_entities.append({
                            'text': entity,
                            'type': entity_type,
                            'position': (start, end)
                        })
            
            # 在同一个句子中创建实体之间的关系
            for i in range(len(sent_entities)):
                for j in range(i + 1, len(sent_entities)):
                    relations.append({
                        'head': sent_entities[i],
                        'tail': sent_entities[j],
                        'context': sent.strip()
                    })
        
        return relations
    
    def process(self, pdf_path: str) -> Dict[str, Any]:
        """处理PDF文件，提取知识图谱元素"""
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?]', '', text)
        
        # Extract entities and relations
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        
        # Tokenize text
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'entities': entities,
            'relations': relations,
            'tokens': tokens,
            'source': 'pdf',
            'path': pdf_path
        }