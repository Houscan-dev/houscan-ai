import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EligibilityClassifier:
    def __init__(self, criteria_dir: str = "./extracted_criteria_gpt"):
        """
        자격 검증 분류기 초기화
        
        Args:
            criteria_dir: 추출된 자격 요건 JSON 파일들이 있는 디렉토리 경로
        """
        self.criteria_dir = criteria_dir
        self.criteria_data = {}  # 공고별 자격 요건을 저장할 딕셔너리
        self._load_all_criteria()
        
    def _load_all_criteria(self):
        """모든 공고의 자격 요건 로드"""
        if os.path.exists(self.criteria_dir):
            for filename in os.listdir(self.criteria_dir):
                if filename.endswith('_criteria.json'):
                    announcement_id = filename.replace('_criteria.json', '')
                    criteria_path = os.path.join(self.criteria_dir, filename)
                    try:
                        with open(criteria_path, 'r', encoding='utf-8') as f:
                            self.criteria_data[announcement_id] = json.load(f)
                            logging.info(f"자격 요건 로드 완료: {announcement_id}")
                    except Exception as e:
                        logging.error(f"Error loading criteria file {filename}: {e}")

    def _calculate_age(self, birth_date: str) -> int:
        """
        주어진 생년월일로 만 나이 계산
        
        Args:
            birth_date: YYYYMMDD 형식의 생년월일
        
        Returns:
            int: 만 나이
        """
        today = datetime.now()
        birth = datetime.strptime(birth_date, '%Y%m%d')
        age = today.year - birth.year
        
        # 생일이 지났는지 체크
        if (today.month, today.day) < (birth.month, birth.day):
            age -= 1
            
        return age

    def check_eligibility(self, announcement_id: str, user_info: Dict) -> Dict:
        """
        사용자의 지원 가능 여부를 검증
        
        Args:
            announcement_id: 공고 ID
            user_info: 사용자 정보 딕셔너리
        
        Returns:
            Dict: {
                "is_eligible": bool,
                "reasons": List[str],
                "score": float
            }
        """
        if announcement_id not in self.criteria_data:
            return {
                "is_eligible": False,
                "reasons": ["해당 공고의 자격 요건 정보를 찾을 수 없습니다."],
                "score": 0.0
            }
            
        criteria = self.criteria_data[announcement_id]
        reasons = []
        total_conditions = 0
        satisfied_conditions = 0
        
        # criteria에서 null이 아닌 모든 기준을 검사
        for key, value in criteria.items():
            if value is None:  # null 값은 건너뛰기
                continue
                
            total_conditions += 1
            is_satisfied = True
            
            if key == 'age_min':
                age = self._calculate_age(user_info['birth_date'])
                if age < value:
                    reasons.append(f"최소 나이 조건을 만족하지 않습니다. (최소 만 {value}세)")
                    is_satisfied = False
                    
            elif key == 'age_max':
                age = self._calculate_age(user_info['birth_date'])
                if age > value:
                    reasons.append(f"최대 나이 조건을 만족하지 않습니다. (최대 만 {value}세)")
                    is_satisfied = False
                    
            elif key == 'marital_status':
                if value == "미혼" and user_info.get('marital_status') != "미혼":
                    reasons.append("미혼자만 지원 가능합니다.")
                    is_satisfied = False
                    
            elif key == 'income_limit_percent':
                income_str = user_info.get('income_range', '0')
                income_str = ''.join(filter(str.isdigit, income_str))
                try:
                    user_income_percent = float(income_str) if income_str else 0
                    if user_income_percent > value:
                        reasons.append(f"소득 기준을 초과합니다. (기준: 도시근로자 월평균소득의 {value}% 이하)")
                        is_satisfied = False
                except ValueError:
                    reasons.append("소득 범위 형식이 올바르지 않습니다.")
                    is_satisfied = False
                    
            elif key == 'total_asset_limit_won':
                if user_info.get('total_assets', 0) > value:
                    reasons.append(f"총자산 기준을 초과합니다. (기준: {value:,}원 이하)")
                    is_satisfied = False
                    
            elif key == 'car_asset_limit_won':
                if user_info.get('car_value', 0) > value:
                    reasons.append(f"차량가액 기준을 초과합니다. (기준: {value:,}원 이하)")
                    is_satisfied = False
                    
            elif key == 'must_be_homeless':
                if value and not user_info.get('is_homeless', False):
                    reasons.append("무주택자만 지원 가능합니다.")
                    is_satisfied = False
            
            if is_satisfied:
                satisfied_conditions += 1
        
        # 최종 신뢰도 점수 계산
        final_score = satisfied_conditions / total_conditions if total_conditions > 0 else 0.0
        
        # 최종 결과 반환
        is_eligible = len(reasons) == 0
        
        if is_eligible:
            reasons.append(f"모든 자격 요건을 충족합니다. ({satisfied_conditions}/{total_conditions} 조건 충족)")
        else:
            reasons.append(f"{satisfied_conditions}/{total_conditions} 조건 충족")
        
        return {
            "is_eligible": is_eligible,
            "reasons": reasons,
            "score": final_score
        }

    def get_all_announcements(self) -> List[str]:
        """모든 공고 목록 반환"""
        return list(self.criteria_data.keys())

# 사용 예시
if __name__ == "__main__":
    # 분류기 초기화
    classifier = EligibilityClassifier()
    
    # 테스트용 사용자 정보
    test_user = {
        "birth_date": "19990101",
        "gender": "남성",
        "university_status": "재학 중",
        "recent_graduate": "예",
        "employed": "아니오",
        "job_seeking": "아니오",
        "household_type": "생계·의료·주거급여 수급자 가구",
        "parents_own_house": "아니요",
        "disability_in_family": "예",
        "application_count": 2,
        "total_assets": 100000000,
        "car_value": 5000000,
        "income_range": "100% 이하",
        "marital_status": "미혼",
        "is_homeless": True
    }
    
    # 모든 공고 목록 확인
    announcements = classifier.get_all_announcements()
    logging.info(f"총 공고 수: {len(announcements)}")
    
    # 각 공고에 대해 자격 검증
    for announcement_id in announcements:
        result = classifier.check_eligibility(announcement_id, test_user)
        logging.info(f"\n공고 ID: {announcement_id}")
        logging.info(f"지원 가능 여부: {'가능' if result['is_eligible'] else '불가능'}")
        logging.info(f"사유: {', '.join(result['reasons'])}")
        logging.info(f"신뢰도 점수: {result['score']:.2f}") 