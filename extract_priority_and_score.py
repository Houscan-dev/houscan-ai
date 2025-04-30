import os
import re
import json
import logging
from typing import Dict, Any, List

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

def generate_gpt_response_priority_score(context, max_tokens=2048):
    system_prompt = "You are a helpful assistant designed to extract specific information from Korean housing announcement documents and output it strictly as a JSON object."
    user_prompt = f"""
다음은 주택 입주자 모집 공고문의 일부 내용입니다. 이 내용을 바탕으로 '순위별 자격요건'과 '가점사항'을 **반드시 JSON 객체**로만 추출해주세요.

요구 항목:
- "priority_criteria": (
    순위별 자격요건이 명확히 제시되어 있지 않으면 null로 반환하세요. 그렇지 않으면 아래와 같이 작성하세요.
    [
      {{
        "priority": "1순위" 또는 "2순위" 또는 "3순위",
        "criteria": [자격요건 설명 문자열 리스트]
      }}, ...
    ]
  )
- "score_items": (
    가점사항이 명확히 제시되어 있지 않으면 null로 반환하세요. 그렇지 않으면 아래와 같이 작성하세요.
    [
      {{
        "priority": "1순위" 또는 "2순위" 또는 "3순위" 또는 "공통",
        "items": [
          {{
            "item": "가점 항목명",
            "score": "점수(정수만, 다른 문자 없이 숫자만)"
          }}, ...
        ]
      }}, ...
    ]
  )

'청년' 관련 내용이 있다면 우선적으로 추출하고, 없다면 전체 기준을 추출하세요.
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

def extract_priority_and_score_for_pdf(pdf_name, output_dir="extracted_priority_and_score"):
    search_queries = [
        "청년 1순위 자격요건", "청년 2순위 자격요건", "청년 3순위 자격요건",
        "청년 가점사항", "가점사항", "순위별 자격요건"
    ]
    n_results_to_fetch = 5
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
            content_cleaned = re.sub(r'<h\d>.*?</h\d>\n?', '', doc, 1).strip()
            if content_cleaned and content_cleaned not in context_text:
                context_text += content_cleaned + "\n\n"
    if not context_text:
        logging.warning(f"No relevant content found for any query in {pdf_name}")
        return False

    gpt_response_content = generate_gpt_response_priority_score(context_text)
    if not gpt_response_content:
        logging.error(f"Failed to get response from GPT API for {pdf_name}")
        return False
    extracted_priority_score = parse_gpt_json_output(gpt_response_content)
    if not extracted_priority_score:
        logging.error(f"Failed to parse priority/score using GPT API for {pdf_name}")
        return False

    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_priority_score.json")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_priority_score, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved priority/score for {pdf_name} to {output_filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving priority/score JSON for {pdf_name}: {e}", exc_info=True)
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
            success = extract_priority_and_score_for_pdf(pdf_name, output_dir="extracted_priority_and_score")
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
