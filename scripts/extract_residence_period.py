import os
import re
import json
import logging
from typing import List, Dict, Any

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

def generate_gpt_response_residence_period(context, max_tokens=512):
    system_prompt = "You are a helpful assistant designed to extract specific information from Korean housing announcement documents and output it strictly as a JSON object."
    user_prompt = f"""
다음은 주택 입주자 모집 공고문의 일부 내용입니다. 이 내용에서 '거주기간'에 대한 정보만을 **반드시 JSON 객체**로 추출해주세요.

요구 항목:
- "residence_period": (거주기간에 대한 설명 전체, 예: "2년, 재계약 4회 가능(최대 10년 거주 가능)" 등. 만약 명확한 정보가 없으면 null로 반환)

JSON 외의 설명은 절대 포함하지 마세요.

[공고문 내용 시작]
{context}
[공고문 내용 끝]

JSON 응답:
"""
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
    return response.choices[0].message.content

def parse_gpt_json_output(gpt_response_content):
    try:
        return json.loads(gpt_response_content)
    except Exception as e:
        logging.error(f"JSON 파싱 오류: {e}")
        return None

def extract_residence_period_for_pdf(pdf_name, output_dir="extracted_residence_period"):
    search_queries = [
        "거주기간", "임대기간", "최대 거주 가능 기간", "재계약 가능 횟수"
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
                context_text += content_cleaned + "\\n\\n"
    if not context_text:
        logging.warning(f"No relevant content found for any query in {pdf_name}")
        return False

    gpt_response_content = generate_gpt_response_residence_period(context_text)
    if not gpt_response_content:
        logging.error(f"Failed to get response from GPT API for {pdf_name}")
        return False
    extracted_residence_period = parse_gpt_json_output(gpt_response_content)
    if not extracted_residence_period:
        logging.error(f"Failed to parse residence period using GPT API for {pdf_name}")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_residence_period.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_residence_period, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved residence period for {pdf_name} to {output_filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving residence period JSON for {pdf_name}: {e}", exc_info=True)
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
            success = extract_residence_period_for_pdf(pdf_name, output_dir="extracted_residence_period")
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