import os
import re
import json
import logging
from typing import List, Dict, Any

import chromadb
import openai
from dotenv import load_dotenv
import time

# 환경설정 및 로깅
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
GPT_MODEL_NAME = "gpt-4-turbo"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="processed_chunks")

# 재처리할 PDF 파일 목록
TARGET_PDFS = [
    "1_ [공고문] 2021년 2차 청년 매입임대주택 입주자모집(2021_12_30_) 공고문_홈페이지 공개용.pdf",
    "첨부_[공고문] 청년 매입임대주택 잔여세대 입주자 모집 공고(2019_10_14_공고).pdf",
    "2020년 청년창업인 임대주택 입주자 모집공고문(재공급).pdf",
    "[마을과집]SH특화형 매입임대주택(청년) 입주자 모집 공고문_20250307.pdf"
]

def get_embedding(text):
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text,
            dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"임베딩 생성 중 오류 발생: {e}")
        return None

def generate_gpt_response_housing_info(context, max_tokens=4096):
    system_prompt = """You are a helpful assistant designed to extract housing information from Korean documents.
You must ALWAYS return a valid JSON object with the following structure EXACTLY:
{
    "housing_info": [
        {
            "name": string or null,
            "address": string or null,
            "district": string or null,
            "total_households": string or null,
            "supply_households": string or null,
            "type": string or null,
            "house_type": string or null,
            "elevator": boolean or null,
            "parking": string or null
        }
    ]
}
Never include any explanations or additional text. Only output the JSON object."""

    user_prompt = f"""다음 주택 공고문에서 주택 정보를 추출하여 JSON 형식으로 반환해주세요.
각 필드가 없는 경우 반드시 null을 사용하세요.
문자열은 반드시 큰따옴표(")로 감싸주세요.
주소에서 자치구(구)를 찾아 district 필드에 넣어주세요.
승강기 여부는 반드시 true/false/null 중 하나여야 합니다.

[공고문 내용]
{context}"""

    try:
        logging.info("GPT API 호출 시작...")
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # 더 결정적인 응답을 위해 temperature를 0으로 설정
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        logging.info(f"GPT API 응답 받음: {content[:200]}...")
        return content
    except Exception as e:
        logging.error(f"GPT API 호출 중 오류 발생: {e}")
        return None

def parse_gpt_json_output(gpt_response_content):
    if not gpt_response_content:
        logging.error("GPT API 응답이 비어있습니다.")
        return None
        
    try:
        # JSON 문자열 정리
        cleaned_content = gpt_response_content.strip()
        if not cleaned_content.startswith('{'):
            logging.error("응답이 JSON 객체로 시작하지 않습니다.")
            return None
            
        # JSON 파싱
        result = json.loads(cleaned_content)
        
        # housing_info 키가 있는지 확인
        if 'housing_info' not in result:
            logging.error("응답에 housing_info 키가 없습니다.")
            return None
            
        # housing_info가 리스트인지 확인
        if not isinstance(result['housing_info'], list):
            logging.error(f"housing_info가 리스트가 아닙니다. 현재 타입: {type(result['housing_info'])}")
            return None
            
        # housing_info가 비어있는지 확인
        if not result['housing_info']:
            logging.warning("housing_info 리스트가 비어있습니다.")
            return None
            
        # 각 항목의 필수 필드 확인
        required_fields = ['name', 'address', 'district', 'total_households', 'supply_households', 
                         'type', 'house_type', 'elevator', 'parking']
        
        for item in result['housing_info']:
            for field in required_fields:
                if field not in item:
                    item[field] = None
            
            # elevator 필드가 boolean 또는 null인지 확인
            if item['elevator'] is not None and not isinstance(item['elevator'], bool):
                item['elevator'] = None
        
        logging.info(f"성공적으로 파싱된 housing_info 배열 길이: {len(result['housing_info'])}")
        return result
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON 파싱 오류: {e}")
        logging.error(f"문제가 있는 JSON 문자열: {gpt_response_content}")
        return None
    except Exception as e:
        logging.error(f"예상치 못한 오류 발생: {e}")
        return None

def extract_housing_info_for_pdf(pdf_name, output_dir="extracted_housing_info"):
    # 공급주택 정보가 있는 섹션의 제목들
    section_titles = [
        "공급주택", "공급현황", "주택정보", "주택현황", "주택", "공급",
        "공급대상", "공급세대", "공급규모", "공급호수", "세대현황",
        "임대주택", "임대조건", "주택공급", "주택유형", "주택규모",
        "건설위치", "공급내역", "공급호실", "세대수", "공급대상주택",
        "주택개요", "주택단지", "단지개요", "단지현황", "공급내용",
        "공급규모 및 대상", "공급대상 및 공급호수", "공급호수 및 공급대상"
    ]
    
    context_text = ""
    max_context_length = 12000  # 컨텍스트 길이 증가
    
    # 각 섹션 제목으로 검색
    for title in section_titles:
        if len(context_text) > max_context_length:
            break
            
        query_embedding = get_embedding(title)
        if query_embedding is None:
            continue
            
        # 제목이 포함된 문서 검색
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,  # 결과 수 증가
            where={"filename": pdf_name},
            include=["documents", "metadatas"]
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            continue
            
        # 해당 섹션의 내용 추출
        for section_content in results['documents'][0]:
            if section_content and section_content not in context_text:
                if len(context_text) + len(section_content) > max_context_length:
                    continue
                context_text += section_content + "\n\n"
    
    if not context_text:
        logging.warning(f"관련 내용을 찾을 수 없습니다: {pdf_name}")
        return False

    logging.info(f"컨텍스트 길이: {len(context_text)} 문자")
    gpt_response_content = generate_gpt_response_housing_info(context_text)
    if not gpt_response_content:
        logging.error(f"GPT API 응답을 받지 못했습니다: {pdf_name}")
        return False
        
    extracted_housing_info = parse_gpt_json_output(gpt_response_content)
    if not extracted_housing_info:
        logging.error(f"주택 정보 파싱에 실패했습니다: {pdf_name}")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_housing_info.json")
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_housing_info, f, ensure_ascii=False, indent=2)
            
        logging.info(f"주택 정보가 저장되었습니다: {output_filename}")
        return True
    except Exception as e:
        logging.error(f"JSON 파일 저장 중 오류 발생: {e}", exc_info=True)
        return False

def main():
    start_time = time.time()
    success_count = 0
    fail_count = 0

    for pdf_name in TARGET_PDFS:
        logging.info(f"공고문 처리 중: {pdf_name}")
        try:
            success = extract_housing_info_for_pdf(pdf_name)
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logging.error(f"{pdf_name} 처리 중 예기치 못한 오류 발생: {e}", exc_info=True)
            fail_count += 1

    elapsed_time = time.time() - start_time
    
    logging.info("=" * 50)
    logging.info(f"처리 완료. 성공: {success_count}, 실패: {fail_count}")
    logging.info(f"총 실행 시간: {elapsed_time:.2f}초")
    logging.info("=" * 50)

if __name__ == "__main__":
    main() 