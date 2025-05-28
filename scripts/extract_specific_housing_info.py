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

def generate_gpt_response_housing_info(context, max_tokens=2048):
    system_prompt = """You are a helpful assistant designed to extract specific information from Korean housing announcement documents and output it strictly as a JSON object. You must ensure all strings are properly escaped and the JSON is valid. Keep the response concise and focused on essential housing information only. DO NOT escape Korean characters.

Key points:
1. Extract ALL available information, even if it appears in different sections of the document
2. For type field, if not explicitly specified but the housing is part of a youth housing announcement, use "청년"
3. For parking field, include ANY information about parking, including fees, availability, and type
"""
    user_prompt = f"""
다음은 주택 입주자 모집 공고문의 일부 내용입니다. 이 내용에서 '공급주택 정보'에 대한 정보만을 **반드시 JSON 객체**로 추출해주세요.

요구사항:
1. 각 주택 정보를 배열의 개별 항목으로 반환해주세요.
2. 동일한 주소지라도 주택형(면적)이 다르거나 공급유형이 다른 경우 별도의 항목으로 구분해주세요.
3. 각 주택 정보는 다음 필드를 포함해야 합니다:
   - name: 주택명 또는 단지명 (없으면 null)
   - address: 도로명 주소 또는 지번 주소 전체 (구, 동, 번지 등 포함)
   - district: 자치구명만 추출 (예: "강남구", "송파구" 등)
   - total_households: 해당 단지의 총 세대수 (단위 "세대" 포함)
   - supply_households: 이번에 공급하는 호수 (단위 "호" 포함)
   - type: 공급유형 (예: "신혼Ⅰ", "청년", "다가구" 등. 청년주택 공고문이면서 별도 유형이 명시되지 않은 경우 "청년"으로 지정)
   - house_type: 주택형 (전용면적, 단위 "㎡" 포함)
   - elevator: 승강기 여부 (있음/없음 정보가 명시된 경우만 true/false, 그 외는 null)
   - parking: 주차 관련 정보 (주차장 유무, 주차비, 운영방식 등 주차 관련 정보 모두 포함. 없으면 null)
4. 정보가 없는 필드는 반드시 null로 반환해주세요.
5. 모든 문자열은 반드시 큰따옴표(")로 감싸주세요.
6. 특수문자가 포함된 경우 반드시 이스케이프 처리해주세요.
7. 숫자로만 된 정보라도 문자열로 처리해주세요.
8. 한글은 이스케이프하지 말고 그대로 출력해주세요.
9. 주차장 정보는 공고문 전체를 검토하여 관련 정보를 모두 포함해주세요.

JSON 형식:
{{
    "housing_info": [
        {{
            "name": "주택명",
            "address": "주소",
            "district": "자치구",
            "total_households": "총 세대수",
            "supply_households": "공급호수",
            "type": "유형",
            "house_type": "주택형",
            "elevator": true/false/null,
            "parking": "주차장 정보"
        }}
    ]
}}

JSON 외의 설명은 절대 포함하지 마세요.

[공고문 내용 시작]
{context}
[공고문 내용 끝]

JSON 응답:
"""
    try:
        logging.info("GPT API 호출 시작...")
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
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
        logging.info(f"정리 전 JSON 문자열: {cleaned_content[:200]}...")
        
        # 불완전한 JSON 문자열 처리
        if not cleaned_content.startswith('{'):
            cleaned_content = '{' + cleaned_content
        if not cleaned_content.endswith('}'):
            last_valid_brace = cleaned_content.rfind('}')
            if last_valid_brace != -1:
                cleaned_content = cleaned_content[:last_valid_brace + 1]
            else:
                cleaned_content = cleaned_content + '}'
            
        logging.info(f"정리 후 JSON 문자열: {cleaned_content[:200]}...")
        
        # JSON 파싱
        result = json.loads(cleaned_content)
        
        # housing_info가 배열인지 확인
        if not isinstance(result.get('housing_info'), list):
            logging.error(f"housing_info가 배열 형식이 아닙니다. 타입: {type(result.get('housing_info'))}")
            return None
            
        # 배열이 비어있지 않은지 확인
        if not result['housing_info']:
            logging.warning("housing_info 배열이 비어있습니다.")
            
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
        "주택개요", "주택단지", "단지개요", "단지현황", "공급내용"
    ]
    
    context_text = ""
    max_context_length = 8000
    
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
            n_results=2,  # 각 섹션당 2개의 결과
            where={
                "$and": [
                    {"filename": pdf_name},
                    {"title": {"$eq": title}}  # 정확한 일치만 사용
                ]
            },
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
        # 제목 매칭이 실패한 경우, 일반적인 검색 시도
        search_queries = [
            "주택명", "주소", "세대수", "공급호수", "유형", "주택형", "승강기", "주차장"
        ]
        
        for query in search_queries:
            if len(context_text) > max_context_length:
                break
                
            query_embedding = get_embedding(query)
            if query_embedding is None:
                continue
                
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                where={"filename": pdf_name},
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                continue
                
            content = results['documents'][0][0]
            if content and content not in context_text:
                if len(context_text) + len(content) > max_context_length:
                    continue
                context_text += content + "\n\n"
    
    if not context_text:
        logging.warning(f"No relevant content found for {pdf_name}")
        return False

    logging.info(f"컨텍스트 길이: {len(context_text)} 문자")
    gpt_response_content = generate_gpt_response_housing_info(context_text)
    if not gpt_response_content:
        logging.error(f"Failed to get response from GPT API for {pdf_name}")
        return False
    extracted_housing_info = parse_gpt_json_output(gpt_response_content)
    if not extracted_housing_info:
        logging.error(f"Failed to parse housing info using GPT API for {pdf_name}")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_housing_info.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_housing_info, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved housing info for {pdf_name} to {output_filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving housing info JSON for {pdf_name}: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # ChromaDB에서 모든 고유한 파일명 가져오기
    try:
        results = collection.get()
        unique_filenames = set(meta['filename'] for meta in results['metadatas'])
        logging.info(f"ChromaDB에서 찾은 파일 목록:")
        for filename in sorted(unique_filenames):
            logging.info(f"- {filename}")
    except Exception as e:
        logging.error(f"ChromaDB에서 파일명 추출 중 오류: {e}", exc_info=True)
        exit()

    # 처리할 특정 파일 목록 (PDF 확장자 포함)
    target_files = [
        "2024년 2차 청년안심주택 공고문-최종(웹용).pdf",
        "2023년 3차 청년안심주택-공고문-웹용(최종).pdf"
    ]

    # 파일 존재 여부 확인
    for pdf_name in target_files:
        if pdf_name not in unique_filenames:
            logging.warning(f"'{pdf_name}'을(를) ChromaDB에서 찾을 수 없습니다.")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for pdf_name in target_files:
        if pdf_name in unique_filenames:
            logging.info(f"공고문 처리 중: {pdf_name}")
            try:
                success = extract_housing_info_for_pdf(pdf_name, output_dir="extracted_housing_info")
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logging.error(f"{pdf_name} 처리 중 예기치 못한 오류 발생: {e}", exc_info=True)
                fail_count += 1
        else:
            logging.error(f"'{pdf_name}'이(가) ChromaDB에 없어 건너뜁니다.")
            fail_count += 1

    elapsed_time = time.time() - start_time

    logging.info("=" * 50)
    logging.info(f"모든 공고문 처리 완료. 성공: {success_count}, 실패: {fail_count}")
    logging.info(f"총 실행 시간: {elapsed_time:.2f}초")
    logging.info("=" * 50) 