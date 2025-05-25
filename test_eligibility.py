import os
import json
from eligibility_checker import check_eligibility

def load_criteria_json(filename):
    """공고문의 기준 데이터를 로드합니다."""
    file_path = os.path.join('data', 'extracted', 'criteria_json', filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 최신 공고문의 기준 데이터 로드
criteria_data = load_criteria_json('2024년_2차_청년안심주택_공고문-최종_웹용__criteria.json')

# 테스트 케이스 1: 모든 조건을 만족하는 2순위 신청자
test_case_1 = {
    "user_data": {
        "age": 25,
        "total_assets": 280000000,  # 2억 8천만원 이하 (2순위 기준)
        "car_assets": 20000000,     # 2,468만원 이하
        "is_homeless": True,
        "marital_status": "미혼",
        "monthly_income": 1800000,   # 70% 이하 (2순위 기준)
        "household_members": 1,
        "nationality": "대한민국",
        "application_type": "청년"
    },
    "criteria_data": criteria_data
}

# 테스트 케이스 2: 1순위 신청자 (수급자)
test_case_2 = {
    "user_data": {
        "age": 28,
        "total_assets": 100000000,  # 1억원
        "car_assets": 0,
        "is_homeless": True,
        "marital_status": "미혼",
        "monthly_income": 1000000,   # 수급자
        "household_members": 1,
        "nationality": "대한민국",
        "application_type": "청년",
        "priority_status": "수급자"
    },
    "criteria_data": criteria_data
}

# 테스트 케이스 3: 부적격 사례 (나이 초과)
test_case_3 = {
    "user_data": {
        "age": 40,
        "total_assets": 280000000,
        "car_assets": 20000000,
        "is_homeless": True,
        "marital_status": "미혼",
        "monthly_income": 1800000,
        "household_members": 1,
        "nationality": "대한민국",
        "application_type": "청년"
    },
    "criteria_data": criteria_data
}

# 테스트 케이스 4: 부적격 사례 (자산 초과)
test_case_4 = {
    "user_data": {
        "age": 25,
        "total_assets": 300000000,  # 3억원 (기준 초과)
        "car_assets": 25000000,     # 2,500만원 (기준 초과)
        "is_homeless": True,
        "marital_status": "미혼",
        "monthly_income": 1800000,
        "household_members": 1,
        "nationality": "대한민국",
        "application_type": "청년"
    },
    "criteria_data": criteria_data
}

def run_test(test_case, case_number):
    print(f"\n테스트 케이스 {case_number} 실행:")
    print("-" * 50)
    eligible, reason = check_eligibility(test_case["user_data"], test_case["criteria_data"])
    print(f"지원 가능 여부: {'가능' if eligible else '불가능'}")
    print(f"판단 근거: {reason}")
    print("-" * 50)

if __name__ == "__main__":
    print("임대주택 지원 자격 검증 테스트 시작")
    print("=" * 50)
    print(f"테스트 대상 공고문: 2024년 2차 청년안심주택")
    print("=" * 50)
    
    # 각 테스트 케이스 실행
    run_test(test_case_1, 1)
    run_test(test_case_2, 2)
    run_test(test_case_3, 3)
    run_test(test_case_4, 4) 