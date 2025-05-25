import os
import json
import logging
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_with_gpt(content):
    """
    GPT를 사용하여 내용을 정리하고 JSON 형식으로 변환
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "주택 입주자 모집 공고문의 자격요건을 명확하고 일관된 형식으로 정리하는 전문가입니다. '~습니다' 문체를 사용합니다."
                },
                {
                    "role": "user",
                    "content": f"다음 자격요건을 깔끔하게 정리해서 하나의 문단으로 작성해주세요:\n\n{content}"
                }
            ],
            temperature=0.1,
            max_tokens=4096
        )
        
        formatted_content = response.choices[0].message.content.strip()
        return formatted_content
    except Exception as e:
        logger.error(f"GPT API 호출 중 오류 발생: {str(e)}")
        return None

def convert_to_json_format(input_dir="data/extracted/criteria3", output_dir="data/extracted/criteria_json"):
    """
    criteria3 디렉토리의 파일들을 JSON 형식으로 변환하여 새로운 디렉토리에 저장
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 입력 디렉토리의 모든 파일 처리
    input_files = [f for f in os.listdir(input_dir) if f.endswith('_criteria.json')]
    logger.info(f"총 {len(input_files)}개의 파일을 처리합니다.")
    
    success_count = 0
    fail_count = 0
    
    for filename in tqdm(input_files, desc="파일 처리 중"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # 파일 내용 읽기
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # GPT로 내용 정리
            formatted_content = format_with_gpt(content)
            if formatted_content is None:
                raise Exception("GPT 처리 실패")
            
            # JSON 형식으로 변환
            json_content = {
                "content": formatted_content
            }
            
            # 새로운 파일로 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
            
            success_count += 1
            logger.debug(f"파일 변환 성공: {filename}")
            
        except Exception as e:
            fail_count += 1
            logger.error(f"파일 처리 중 오류 발생 ({filename}): {str(e)}")
    
    logger.info("=" * 50)
    logger.info(f"처리 완료. 성공: {success_count}, 실패: {fail_count}")
    logger.info("=" * 50)

if __name__ == "__main__":
    convert_to_json_format() 