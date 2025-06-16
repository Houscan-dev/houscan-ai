# -*- coding: utf-8 -*-
import json
from typing import Dict, Any
from openai import OpenAI
import os
from dotenv import load_dotenv

# .env 파일 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# criteria 파일 경로로부터 대응되는 priority_score 파일 경로 생성
# 예: .../criteria/공고명_criteria.json → .../priority_score/공고명_priority_score.json
def get_priority_score_path(criteria_file: str) -> str:
    base_name = os.path.basename(criteria_file)
    priority_score_file = base_name.replace("_criteria.json", "_priority_score.json")
    priority_score_dir = os.path.join(os.path.dirname(os.path.dirname(criteria_file)), "priority_score")
    return os.path.join(priority_score_dir, priority_score_file)

# 입력 데이터 필드 설명 (프롬프트에 공통 포함)
field_description = '''
[입력 데이터 필드 설명]
- id: 사용자 구별용 id
- age: 나이
- birth_date: 생년월일 (YYMMDD)
- gender: 성별 (M: 남성, F: 여성)
- university: 대학생 재학중인지 여부 (true/false)
- graduate: 대학 또는 고등학교를 졸업한지 2년 이내인지 여부 (true/false)
- employed: 직장 재직중인지 여부 (true/false)
- job_seeker: 취업준비생 여부 (true/false)
- welfare_receipient: 생계, 의료, 주거급여 수급자 가구, 지원대상 한부모 가족, 차상위계층 가구 중 해당사항이 있는지 여부 (true/false)
- parents_own_house: 부모가 무주택자인지 여부 (true/false)
- disability_in_family: 자신이나 가구원 중에 본인 명의의 장애인 등록증을 소유하고 있는 사람이 있는지 여부 (true/false)
- subscription_account: 청약 납입 횟수
- total_assets: 총 자산 (원 단위)
- car_value: 소유하고 있는 자동차 가액 (원 단위)
- income_range: 가구당 월평균 소득 구간 (예: "100% 이하")
- create_at: 계정 생성 날짜 (ISO 8601 형식)
- user: 사용자 구별 id (중복 가능)
'''

# 2. 우선순위 판단 (priority_criteria를 활용)
def check_priority_with_llm(user_data: Dict[str, Any], priority_data: Dict) -> dict:
    """
    LLM을 사용하여 우선순위만 판단하고, 판단 이유도 반환
    - priority_criteria의 각 순위별 criteria 중 하나라도 만족하면 해당 순위로 인정
    - 여러 순위에 모두 해당될 경우, priority_criteria 배열에서 더 앞에 있는(더 높은) 순위를 최종 우선순위로 판단
    - 어떤 순위에도 해당하지 않으면 "우선순위 해당없음"으로 판단
    """
    priority_prompt = f"""
{field_description}

[우선순위 기준]
{json.dumps(priority_data["priority_criteria"], ensure_ascii=False, indent=2)}

[사용자 정보]
{json.dumps(user_data, ensure_ascii=False, indent=2)}

위 우선순위 기준에 따라 해당 사용자의 우선순위를 판단해주세요.
- priority_criteria의 각 순위별 criteria 중 하나라도 만족하면 해당 순위로 인정
- 여러 순위에 모두 해당될 경우, priority_criteria 배열에서 더 앞에 있는(더 높은) 순위를 최종 우선순위로 판단
- 어떤 순위에도 해당하지 않으면 "우선순위 해당없음"으로 판단

다음과 같은 JSON 형식으로, 반드시 JSON만 반환하세요. 다른 설명이나 텍스트는 절대 포함하지 마세요:
{{
    "priority": "판단된 우선순위",
    "reason": "판단 이유를 간단히 서술"
}}
"""
    priority_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "당신은 주택 신청 우선순위를 판단하는 전문가입니다. 주어진 우선순위 기준에 따라 정확하게 판단해주세요."},
            {"role": "user", "content": priority_prompt}
        ],
        temperature=0
    )
    try:
        priority_result = json.loads(priority_response.choices[0].message.content)
        return priority_result
    except (json.JSONDecodeError, KeyError):
        return {"priority": "우선순위 판단 오류", "reason": "판단 오류"}

# 3. 신청자격 판단 (우선순위 결과를 참고정보로 활용)
def check_eligibility_with_llm(user_data: Dict[str, Any], criteria_str: str, priority_result: dict) -> dict:
    """
    LLM을 사용하여 자격만 판단하고, 판단 이유도 반환
    - 우선순위 판단 결과(priority_result)를 참고정보로 프롬프트에 포함
    """
    eligibility_prompt = f"""
{field_description}

[신청자격 요건]
{criteria_str}

[사용자 정보]
{json.dumps(user_data, ensure_ascii=False, indent=2)}

[참고 우선순위 정보]
{json.dumps(priority_result, ensure_ascii=False, indent=2)}

신청자격 요건의 모든 조건을 검토하여 true/false로 판단하고, 그 이유도 함께 설명해주세요.

다음과 같은 JSON 형식으로, 반드시 JSON만 반환하세요. 다른 설명이나 텍스트는 절대 포함하지 마세요:
{{
    "is_eligible": true/false,
    "reason": "판단 이유를 간단히 서술"
}}
"""
    eligibility_response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "당신은 주택 신청 자격을 판단하는 전문가입니다. 주어진 공고문의 기준에 따라 정확하게 판단해주세요."},
            {"role": "user", "content": eligibility_prompt}
        ],
        temperature=0
    )
    try:
        eligibility_result = json.loads(eligibility_response.choices[0].message.content)
        return eligibility_result
    except (json.JSONDecodeError, KeyError):
        return {"is_eligible": False, "reason": "판단 오류"}

# 1~4. 전체 흐름을 담당하는 메인 함수
def process_test_cases(test_cases_file: str, criteria_file: str) -> Dict[str, Dict[str, Any]]:
    """
    1. 각 공고문에 해당하는 criteria.json과 priority_score.json의 priority_criteria 항목을 가져옴
    2. priority_criteria를 활용해서 우선순위 먼저 판단
    3. 2번 단계에서 얻은 우선순위 정보를 신청가능여부 판단 프롬프트에 다시 활용해서, 신청자격 해당됨/해당안됨 여부 판단
    4. 신청자격, 우선순위를 최종적으로 종합해서 json으로 출력
    """
    # 파일 읽기 (테스트 케이스, 기준, 우선순위 기준)
    with open(test_cases_file, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
    with open(criteria_file, 'r', encoding='utf-8') as f:
        criteria_data = json.load(f)
        criteria_str = criteria_data.get('content', '')
    priority_score_file = get_priority_score_path(criteria_file)
    with open(priority_score_file, 'r', encoding='utf-8') as f:
        priority_data = json.load(f)

    results = {}
    # 각 테스트 케이스에 대해 2→3번 순서로 판단, 결과를 합쳐서 저장
    for case_id, case_data in test_cases.items():
        # 2. 우선순위 먼저 판단
        priority_result = check_priority_with_llm(case_data["user_data"], priority_data)
        # 3. 우선순위 결과를 참고하여 자격 판단
        eligibility_result = check_eligibility_with_llm(case_data["user_data"], criteria_str, priority_result)
        # 4. 결과 종합
        if not eligibility_result["is_eligible"]:
            results[case_id] = {
                "is_eligible": False,
                "priority": priority_result.get("priority", "해당없음"),
                "eligibility_reason": eligibility_result.get("reason", ""),
                "priority_reason": priority_result.get("reason", "")
            }
        else:
            results[case_id] = {
                "is_eligible": True,
                "priority": priority_result.get("priority", ""),
                "eligibility_reason": eligibility_result.get("reason", ""),
                "priority_reason": priority_result.get("reason", "")
            }
    return results

# 메인 실행부: 파일 경로 지정 및 실행
if __name__ == "__main__":
    # 예시 실행
    test_cases_file = "test_cases.json"
    criteria_file = os.path.join("data", "extracted", "criteria", "2020년_청년창업인_임대주택_입주자_모집공고_criteria.json")
    results = process_test_cases(test_cases_file, criteria_file)
    print(json.dumps(results, ensure_ascii=False, indent=2)) 