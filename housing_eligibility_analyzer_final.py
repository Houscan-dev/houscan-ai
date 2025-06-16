import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from django.conf import settings
from users.models import User
from .models import AnnouncementDocument, Announcement
from django.db import models
import time
from openai import RateLimitError

# .env 파일 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 데이터베이스에서 사용자 데이터 조회
def get_user_data_from_db(user_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Django ORM을 사용하여 사용자 데이터를 조회하여 반환
    user_ids: 특정 사용자들만 조회하고 싶을 때 사용 (None이면 모든 사용자)
    """
    from profiles.models import Profile  # 여기서 import
    
    try:
        print(f"\n=== get_user_data_from_db 호출 (user_ids: {user_ids}) ===")
        if user_ids:
            profiles = Profile.objects.filter(user_id__in=user_ids)
        else:
            profiles = Profile.objects.all()
        
        print(f"조회된 프로필 수: {profiles.count()}")
        
        # 데이터 형식을 기존 test_cases 형태로 변환
        user_data = {}
        for profile in profiles:
            print(f"프로필 데이터 처리 중: user_{profile.user.id}")
            user_data[f"user_{profile.user.id}"] = {
                "id": profile.user.id,
                "age": profile.age,
                "birth_date": profile.birth_date,
                "gender": profile.gender,
                "university": profile.university,
                "graduate": profile.graduate,
                "employed": profile.employed,
                "job_seeker": profile.job_seeker,
                "welfare_receipient": profile.welfare_receipient,
                "parents_own_house": profile.parents_own_house,
                "disability_in_family": profile.disability_in_family,
                "subscription_account": profile.subscription_account,
                "total_assets": profile.total_assets,
                "car_value": profile.car_value,
                "income_range": profile.income_range,
                "create_at": profile.create_at.isoformat(),
                "user": profile.user.id
            }
        
        print(f"반환할 사용자 데이터: {list(user_data.keys())}")
        print("=== get_user_data_from_db 완료 ===\n")
        return user_data
        
    except Exception as e:
        print(f"데이터 조회 오류: {e}")
        return {}

# criteria 파일 경로로부터 대응되는 priority_score 파일 경로 생성
def get_priority_score_path(criteria_file: str) -> str:
    """criteria 파일 경로로부터 대응되는 priority_score 파일 경로를 생성"""
    try:
        # criteria 파일의 기본 이름
        base_name = os.path.basename(criteria_file)
        
        # AnnouncementDocument에서 해당 공고의 priority_score 문서 찾기
        try:
            # criteria 문서 찾기
            criteria_doc = AnnouncementDocument.objects.get(data_file__endswith=base_name)
            
            # 해당 공고의 priority_score 문서 찾기
            priority_doc = AnnouncementDocument.objects.get(
                announcement_id=criteria_doc.announcement_id,
                doc_type='priority_score'
            )
            return priority_doc.data_file.path
            
        except AnnouncementDocument.DoesNotExist as e:
            raise FileNotFoundError(f"우선순위 점수 파일을 찾을 수 없습니다: {str(e)}")
        
    except Exception as e:
        raise Exception(f"우선순위 점수 파일 경로 생성 중 오류 발생: {str(e)}")

# 입력 데이터 필드 설명 (프롬프트에 공통 포함)
field_description = '''
[입력 데이터 필드]
- id: 사용자 ID
- age: 나이
- birth_date: 생년월일 (YYMMDD)
- gender: 성별 (M/F)
- university: 대학생 재학여부
- graduate: 졸업 2년 이내 여부
- employed: 재직여부
- job_seeker: 취업준비생 여부
- welfare_receipient: 수급자/한부모/차상위 여부
- parents_own_house: 부모 무주택 여부
- disability_in_family: 장애인 가구원 여부
- subscription_account: 청약 납입 횟수
- total_assets: 총 자산 (원)
- car_value: 자동차 가액 (원)
- income_range: 가구당 월평균 소득 구간
'''

# 2. 우선순위 판단 (priority_criteria를 활용)
def check_priority_with_llm(user_data: Dict[str, Any], priority_data: Dict) -> dict:
    """
    LLM을 사용하여 우선순위만 판단
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
    "priority": "판단된 우선순위"
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "당신은 주택 신청 우선순위를 판단하는 전문가입니다. 주어진 우선순위 기준에 따라 정확하게 판단해주세요."},
                {"role": "user", "content": priority_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        if "insufficient_quota" in str(e):
            return {
                "priority": "처리 오류",
                "reasons": ["OpenAI API 할당량이 초과되었습니다. 잠시 후 다시 시도해주세요."]
            }
        return {
            "priority": "처리 오류",
            "reasons": [f"처리 중 오류 발생: {str(e)}"]
        }

# 3. 신청자격 판단 (우선순위 결과를 참고정보로 활용)
def check_eligibility_with_llm(user_data: Dict[str, Any], criteria_str: str, priority_result: dict) -> dict:
    """
    LLM을 사용하여 자격만 판단
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

신청자격 요건의 모든 조건을 검토하여 true/false로 판단해주세요.
자격이 없는 경우, 충족하지 못한 모든 조건과 사유를 한국어로 간단히 나열해주세요.
계정 생성일(create_at)은 신청자격 판단에 사용하지 마세요.

다음과 같은 JSON 형식으로, 반드시 JSON만 반환하세요. 다른 설명이나 텍스트는 절대 포함하지 마세요:
{{
    "is_eligible": true/false,
    "reasons": ["미달 사유1", "미달 사유2"]
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "당신은 주택 신청 자격을 판단하는 전문가입니다. 계정 생성일은 판단 기준에서 제외해주세요. 주어진 공고문의 기준에 따라 정확하게 판단해주세요."},
                {"role": "user", "content": eligibility_prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        if "insufficient_quota" in str(e):
            return {
                "is_eligible": False,
                "reasons": ["OpenAI API 할당량이 초과되었습니다. 잠시 후 다시 시도해주세요."]
            }
        return {
            "is_eligible": False,
            "reasons": [f"처리 중 오류 발생: {str(e)}"]
        }

def analyze_user_eligibility(user_id: str) -> Dict[int, Dict[str, Any]]:
    """
    로그인한 사용자에 대해 모든 공고의 자격을 분석
    Args:
        user_id: 분석할 사용자 ID
    Returns:
        Dict[int, Dict[str, Any]]: 공고 ID를 키로 하는 분석 결과
    """
    print(f"\n=== analyze_user_eligibility 시작 (사용자 ID: {user_id}) ===")
    
    # 1. 사용자 데이터 조회 (한 번만)
    print("\n[1] 사용자 데이터 조회")
    user_data = get_user_data_from_db([user_id])
    if not user_data:
        print("사용자 데이터가 없습니다.")
        return {}
        
    user_case = user_data.get(f'user_{user_id}')
    if not user_case:
        print("사용자 케이스가 없습니다.")
        return {}
    print("[1] 사용자 데이터 조회 완료")
    
    # 2. 모든 공고 조회
    print("\n[2] 공고 목록 조회")
    announcements = Announcement.objects.all()
    print(f"조회된 공고 수: {announcements.count()}")
    
    # 3. 모든 공고의 criteria와 priority_score를 한 번에 로드
    print("\n[3] 공고 파일 로드 시작")
    criteria_data = {}
    priority_data = {}
    
    for announcement in announcements:
        try:
            print(f"\n공고 ID {announcement.id} 파일 로드 중...")
            criteria_doc = announcement.documents.get(doc_type="criteria")
            if not os.path.exists(criteria_doc.data_file.path):
                print(f"공고 ID {announcement.id}: criteria 파일이 존재하지 않습니다.")
                continue
                
            with open(criteria_doc.data_file.path, 'r', encoding='utf-8') as f:
                criteria_data[announcement.id] = json.load(f).get('content', '')
            
            priority_score_file = get_priority_score_path(criteria_doc.data_file.path)
            with open(priority_score_file, 'r', encoding='utf-8') as f:
                priority_data[announcement.id] = json.load(f)
            print(f"공고 ID {announcement.id}: 파일 로드 완료")
                
        except Exception as e:
            print(f"공고 ID {announcement.id}: 파일 로드 중 오류 발생: {str(e)}")
            continue
    print("[3] 공고 파일 로드 완료")

    # 4. 각 공고에 대해 분석 수행
    print("\n[4] 공고별 분석 시작")
    results = {}
    for announcement in announcements:
        try:
            print(f"\n공고 ID {announcement.id} 분석 중...")
            if announcement.id not in criteria_data or announcement.id not in priority_data:
                print(f"공고 ID {announcement.id}: 필요한 파일이 없어 분석을 건너뜁니다.")
                continue

            # 자격 판단
            print(f"공고 ID {announcement.id}: 자격 판단 시작")
            try:
                eligibility_result = check_eligibility_with_llm(user_case, criteria_data[announcement.id], {})
            except Exception as e:
                print(f"공고 ID {announcement.id}: check_eligibility_with_llm 호출 중 OpenAI API 에러: {str(e)}")
                results[announcement.id] = {
                    "is_eligible": False,
                    "priority": "처리 오류",
                    "reasons": [f"OpenAI API 호출 오류: {str(e)}"]
                }
                continue

            if not eligibility_result.get("is_eligible", False):
                print(f"공고 ID {announcement.id}: 자격 미달")
                results[announcement.id] = {
                    "is_eligible": False,
                    "priority": "해당없음",
                    "reasons": eligibility_result.get("reasons", ["자격 요건을 충족하지 못함"])
                }
                continue

            # 우선순위 판단
            print(f"공고 ID {announcement.id}: 우선순위 판단 시작")
            try:
                priority_result = check_priority_with_llm(user_case, priority_data[announcement.id])
            except Exception as e:
                print(f"공고 ID {announcement.id}: check_priority_with_llm 호출 중 OpenAI API 에러: {str(e)}")
                results[announcement.id] = {
                    "is_eligible": True,
                    "priority": "처리 오류",
                    "reasons": [f"OpenAI API 호출 오류: {str(e)}"]
                }
                continue

            results[announcement.id] = {
                "is_eligible": True,
                "priority": priority_result.get("priority", ""),
                "reasons": ["자격 요건을 모두 충족함"]
            }
            print(f"공고 ID {announcement.id}: 분석 완료")

        except Exception as e:
            print(f"공고 ID {announcement.id}: 분석 중 오류 발생: {str(e)}")
            results[announcement.id] = {
                "is_eligible": False,
                "priority": "처리 오류",
                "reasons": [f"처리 중 오류 발생: {str(e)}"]
            }
    
    print("\n=== analyze_user_eligibility 완료 ===")
    return results 