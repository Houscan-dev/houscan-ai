"""
공고문의 유의사항을 추출하는 스크립트
"""
import os
import json
import re
import logging
from typing import Dict, Any, List
import PyPDF2
from pathlib import Path

import chromadb
from transformers import AutoTokenizer, AutoModel
import torch
import openai
from dotenv import load_dotenv
import time

# 환경설정 및 로깅
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
embedding_model = AutoModel.from_pretrained("BAAI/bge-m3")
if torch.cuda.is_available():
    embedding_model.to("cuda")
embedding_model.eval()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="processed_chunks")

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
GPT_MODEL_NAME = "gpt-3.5-turbo"

def get_embedding(text):
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    device = embedding_model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.cpu().squeeze().tolist()

def generate_gpt_response_precautions(context, max_tokens=2048):
    system_prompt = "You are a helpful assistant designed to extract specific information from Korean housing announcement documents and output it strictly as a JSON object. You must ensure all strings are properly escaped and the JSON is valid. Keep the response concise and focused on essential precautions only."
    user_prompt = f"""
다음은 주택 입주자 모집 공고문의 일부 내용입니다. 이 내용에서 '유의사항'에 대한 정보만을 **반드시 JSON 객체**로 추출해주세요.

요구사항:
1. 각 유의사항을 배열의 개별 항목으로 반환해주세요.
2. 각 유의사항은 순수한 문장으로 작성해주세요.
3. 소제목이나 번호는 제거하고, 각 항목을 문장으로 변환해주세요.
4. 중복되는 내용은 제거해주세요.
5. 만약 명확한 유의사항이 없으면 빈 배열([])로 반환해주세요.
6. 모든 문자열은 반드시 큰따옴표(")로 감싸주세요.
7. 특수문자가 포함된 경우 반드시 이스케이프 처리해주세요.
8. 각 유의사항은 간단명료하게 작성해주세요.

JSON 형식:
{{
    "precautions": [
        "첫 번째 유의사항",
        "두 번째 유의사항",
        "세 번째 유의사항"
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
        logging.info(f"GPT API 응답 받음: {content[:200]}...")  # 응답의 처음 200자만 로깅
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
        logging.info(f"정리 전 JSON 문자열: {cleaned_content[:200]}...")  # 정리 전 문자열의 처음 200자만 로깅
        
        # 불완전한 JSON 문자열 처리
        if not cleaned_content.startswith('{'):
            cleaned_content = '{' + cleaned_content
        if not cleaned_content.endswith('}'):
            # 마지막 유효한 JSON 객체를 찾아서 자르기
            last_valid_brace = cleaned_content.rfind('}')
            if last_valid_brace != -1:
                cleaned_content = cleaned_content[:last_valid_brace + 1]
            else:
                cleaned_content = cleaned_content + '}'
            
        logging.info(f"정리 후 JSON 문자열: {cleaned_content[:200]}...")  # 정리 후 문자열의 처음 200자만 로깅
        
        # JSON 파싱
        result = json.loads(cleaned_content)
        
        # precautions가 배열인지 확인
        if not isinstance(result.get('precautions'), list):
            logging.error(f"precautions가 배열 형식이 아닙니다. 타입: {type(result.get('precautions'))}")
            return None
            
        # 배열이 비어있지 않은지 확인
        if not result['precautions']:
            logging.warning("precautions 배열이 비어있습니다.")
            
        logging.info(f"성공적으로 파싱된 precautions 배열 길이: {len(result['precautions'])}")
        return result
    except json.JSONDecodeError as e:
        logging.error(f"JSON 파싱 오류: {e}")
        logging.error(f"문제가 있는 JSON 문자열: {gpt_response_content}")
        return None
    except Exception as e:
        logging.error(f"예상치 못한 오류 발생: {e}")
        return None

def extract_precautions_for_pdf(pdf_name, output_dir="extracted_precautions"):
    search_queries = [
        "유의사항", "꼭 확인", "주의사항", "필독", "알림"
    ]
    n_results_to_fetch = 3
    context_text = ""
    for query in search_queries:
        query_embedding = get_embedding(query)
        if query_embedding is None:
            continue
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_to_fetch,
            where={"filename": pdf_name},
            include=["documents", "metadatas"]
        )
        if not results or not results.get('documents') or not results['documents'][0]:
            continue
        retrieved_docs = results['documents'][0]
        for doc in retrieved_docs:
            content_cleaned = re.sub(r'<h\\d>.*?</h\\d>\\n?', '', doc, 1).strip()
            if content_cleaned and content_cleaned not in context_text:
                context_text += content_cleaned + "\n\n"
    if not context_text:
        logging.warning(f"No relevant content found for any query in {pdf_name}")
        return False

    gpt_response_content = generate_gpt_response_precautions(context_text)
    if not gpt_response_content:
        logging.error(f"Failed to get response from GPT API for {pdf_name}")
        return False
    extracted_precautions = parse_gpt_json_output(gpt_response_content)
    if not extracted_precautions:
        logging.error(f"Failed to parse precautions using GPT API for {pdf_name}")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_precautions.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_precautions, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved precautions for {pdf_name} to {output_filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving precautions JSON for {pdf_name}: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # ChromaDB에서 모든 고유한 파일명 가져오기
    try:
        results = collection.get()
        unique_filenames = set(meta['filename'] for meta in results['metadatas'])
        logging.info(f"총 {len(unique_filenames)}개의 공고문을 찾았습니다.")
    except Exception as e:
        logging.error(f"ChromaDB에서 파일명 추출 중 오류: {e}", exc_info=True)
        exit()

    start_time = time.time()

    success_count = 0
    fail_count = 0

    for pdf_name in unique_filenames:
        logging.info(f"공고문 처리 중: {pdf_name}")
        try:
            success = extract_precautions_for_pdf(pdf_name, output_dir="extracted_precautions")
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logging.error(f"{pdf_name} 처리 중 예기치 못한 오류 발생: {e}", exc_info=True)
            fail_count += 1

    elapsed_time = time.time() - start_time

    logging.info("=" * 50)
    logging.info(f"모든 공고문 처리 완료. 성공: {success_count}, 실패: {fail_count}")
    logging.info(f"총 실행 시간: {elapsed_time:.2f}초")
    logging.info("=" * 50) 