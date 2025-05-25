#사용자의 우선순위와 지원가능여부 판단
import os
import json
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_user_info_from_sql(user_id: str) -> Dict[str, Any]:
    """
    SQL DB에서 사용자 정보를 가져옵니다.
    실제 구현시에는 DB 연결 및 쿼리 로직이 필요합니다.
    
    Args:
        user_id: 사용자 ID
        
    Returns:
        Dict[str, Any]: 사용자 정보
    """
    # 실제 구현시에는 여기에 DB 연결 및 쿼리 로직이 들어갑니다
    # 현재는 예시 데이터를 반환
    return {
        "name": "",
        "birth_date": "",
        "address": "",
        "household_members": 0,
        "income": 0,
        "assets": 0
    }

def determine_priority(user_data: Dict[str, Any], priority_score_path: str) -> Optional[str]:
    """
    사용자의 우선순위를 판단합니다.
    
    Args:
        user_data: 사용자 정보
        priority_score_path: 우선순위 기준 JSON 파일 경로
    
    Returns:
        Optional[str]: 해당하는 우선순위 ("1순위", "2순위" 등). 해당 없으면 None
    """
    try:
        # 우선순위 기준 로드
        with open(priority_score_path, 'r', encoding='utf-8') as f:
            priority_data = json.load(f)
            
        # OpenAI API에 전송할 프롬프트 구성
        prompt = f"""
        신청자의 우선순위를 판단해주세요.

        [우선순위 기준]
        {json.dumps(priority_data["priority_criteria"], ensure_ascii=False, indent=2)}

        [신청자 정보]
        {json.dumps(user_data, ensure_ascii=False, indent=2)}

        위 기준과 신청자 정보를 비교하여 다음 JSON 형식으로 응답해주세요:
        {{"priority": "1순위" 또는 "2순위" 또는 "3순위" 또는 "4순위" 또는 null}}

        * 판단 시 주의사항:
        1. 각 순위의 모든 기준을 충족해야 해당 순위로 판단합니다.
        2. 어떤 순위의 기준도 충족하지 못하면 null을 반환합니다.
        3. 가장 높은 해당 순위를 반환합니다.
        """

        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "임대주택 지원자의 우선순위를 판단하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # API 응답 파싱
        result = json.loads(response.choices[0].message.content)
        return result["priority"]
        
    except Exception as e:
        print(f"우선순위 판단 중 오류 발생: {str(e)}")
        return None

def check_eligibility(user_id: str, criteria_file_path: str, priority_score_path: str) -> Tuple[bool, str]:
    """
    사용자 데이터와 지원 기준을 비교하여 지원 가능 여부와 우선순위를 판단합니다.
    
    Args:
        user_id: 사용자 ID
        criteria_file_path: 지원 기준 JSON 파일 경로
        priority_score_path: 우선순위 기준 JSON 파일 경로
    
    Returns:
        Tuple[bool, str]: (지원 가능 여부, 우선순위)
        우선순위가 없는 경우 빈 문자열(""), 부적격인 경우 "해당없음" 반환
    """
    try:
        # 사용자 정보 가져오기
        user_data = get_user_info_from_sql(user_id)
        
        # 1. 우선순위 판단
        user_priority = ""  # 기본값: 우선순위 없음
        has_priority_criteria = False
        
        try:
            with open(priority_score_path, 'r', encoding='utf-8') as f:
                priority_data = json.load(f)
                if priority_data.get("priority_criteria"):
                    has_priority_criteria = True
                    user_priority = determine_priority(user_data, priority_score_path) or ""
        except (FileNotFoundError, json.JSONDecodeError):
            # 우선순위 파일이 없거나 형식이 잘못된 경우 우선순위 없이 진행
            pass
            
        # 2. 지원 기준 정보 로드
        with open(criteria_file_path, 'r', encoding='utf-8') as f:
            criteria_data = json.load(f)
        
        # 3. 우선순위 정보를 포함한 프롬프트 구성
        priority_info = ""
        check_instruction = ""
        
        if has_priority_criteria:
            if user_priority:
                priority_info = f"\n[신청자 우선순위]\n{user_priority}"
                check_instruction = f"신청자가 {user_priority}에 해당하며 해당 순위의 자격요건을 모두 충족하는지 확인해주세요."
            else:
                priority_info = "\n[신청자 우선순위]\n해당순위 없음"
                check_instruction = "신청자가 어떤 순위에도 해당하지 않으므로 부적격 처리해주세요."
        else:
            check_instruction = "신청자가 자격요건을 모두 충족하는지 확인해주세요."
        
        prompt = f"""
        임대주택 지원자의 자격을 검토해주세요.

        [지원 기준]
        {criteria_data["content"]}

        [신청자 정보]
        {json.dumps(user_data, ensure_ascii=False, indent=2)}{priority_info}

        위 기준과 신청자 정보를 비교하여 다음 JSON 형식으로 응답해주세요:
        {{"eligible": true/false}}

        * 판단 시 주의사항:
        1. 공고문에 명시된 모든 기본 자격 요건을 충족해야 합니다.
        2. {check_instruction}
        3. 부적격 사유가 있다면 즉시 false를 반환합니다.
        4. 판단이 애매한 경우 공고문의 기준을 우선으로 적용해주세요.
        """

        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "임대주택 지원 자격을 검토하고 JSON 형식으로 응답하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # 4. API 응답 파싱 및 최종 결과 반환
        result = json.loads(response.choices[0].message.content)
        is_eligible = result["eligible"]
        
        if is_eligible:
            return True, user_priority
        else:
            return False, "해당없음"
    
    except Exception as e:
        print(f"자격 검토 중 오류 발생: {str(e)}")
        return False, "해당없음" 