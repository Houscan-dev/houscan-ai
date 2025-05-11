"""
문서 파싱 관련 핵심 기능
"""
from typing import Dict, Any, List
import PyPDF2
import os

class DocumentParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt']

    def parse_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서를 파싱하여 구조화된 데이터로 변환
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_formats:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")

        if file_ext == '.pdf':
            return self._parse_pdf(file_path)
        elif file_ext == '.txt':
            return self._parse_txt(file_path)

    def _parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        PDF 파일 파싱
        """
        result = {
            'content': [],
            'metadata': {}
        }
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            result['metadata']['pages'] = len(pdf_reader.pages)
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                result['content'].append(text)

        return result

    def _parse_txt(self, file_path: str) -> Dict[str, Any]:
        """
        텍스트 파일 파싱
        """
        result = {
            'content': [],
            'metadata': {}
        }
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            result['content'].append(content)
            result['metadata']['lines'] = len(content.split('\n'))

        return result 