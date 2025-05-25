#사용자가 지원가능한지 불가능한지 판단
import os
import json
from typing import Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def check_eligibility(user_data: Dict[str, Any], criteria_data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    사용자 데이터와 지원 기준을 비교하여 지원 가능 여부를 판단합니다.
    
    Args:
        user_data: 사용자 정보 딕셔너리
        criteria_data: 지원 기준 정보 딕셔너리
    
    Returns:
        Tuple[bool, str]: (지원 가능 여부, 판단 근거 설명)
    """
    
    # OpenAI API에 전송할 프롬프트 구성
    prompt = f"""
    임대주택 지원자의 자격을 검토해주세요.

    [지원 기준]
    {criteria_data["content"]}

    [신청자 정보]
    {json.dumps(user_data, ensure_ascii=False, indent=2)}

    위 기준과 신청자 정보를 비교하여 다음 JSON 형식으로 응답해주세요:
    {{"eligible": true/false, "reason": "판단 근거를 상세히 설명해주세요"}}

    * 판단 시 주의사항:
    1. 공고문에 명시된 모든 기본 자격 요건을 충족해야 합니다.
    2. 신청자의 순위(1,2,3순위)를 판단하고, 해당 순위의 자격요건을 모두 충족하는지 확인해주세요.
    3. 부적격 사유가 있다면 모두 나열해주세요.
    4. 판단이 애매한 경우 공고문의 기준을 우선으로 적용해주세요.
    """

    try:
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "임대주택 지원 자격을 검토하고 JSON 형식으로 응답하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        # API 응답 파싱
        result = json.loads(response.choices[0].message.content)
        return result["eligible"], result["reason"]
    
    except Exception as e:
        return False, f"자격 검토 중 오류가 발생했습니다: {str(e)}" 