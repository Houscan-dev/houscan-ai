"""
자격요건 분석 서비스
"""
from typing import Dict, Any, List
from core.document_parser import DocumentParser

class EligibilityService:
    def __init__(self):
        self.document_parser = DocumentParser()

    async def analyze_eligibility(self, document_path: str) -> Dict[str, Any]:
        """
        문서를 분석하여 자격요건을 추출
        """
        try:
            # 문서 파싱
            parsed_doc = self.document_parser.parse_document(document_path)
            
            # TODO: 자격요건 분석 로직 구현
            # 1. 텍스트 전처리
            # 2. 자격요건 추출
            # 3. 점수 계산
            
            return {
                "status": "success",
                "eligibility_criteria": [],
                "score": 0,
                "details": {}
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        """
        # TODO: 텍스트 전처리 로직 구현
        return text

    def _extract_criteria(self, text: str) -> List[Dict[str, Any]]:
        """
        자격요건 추출
        """
        # TODO: 자격요건 추출 로직 구현
        return []

    def _calculate_score(self, criteria: List[Dict[str, Any]]) -> float:
        """
        자격요건 점수 계산
        """
        # TODO: 점수 계산 로직 구현
        return 0.0 