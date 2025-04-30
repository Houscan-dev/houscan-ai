import os
import json
import datetime
import re
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import time
from dotenv import load_dotenv
import logging
import openai
import torch

# .env 파일 로드
load_dotenv()

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 임베딩 모델 및 토크나이저 로드 ---
try:
    embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    embedding_model = AutoModel.from_pretrained("BAAI/bge-m3")
    if torch.cuda.is_available():
        embedding_model.to("cuda")
        logging.info("Embedding model moved to GPU.")
    else:
        logging.info("CUDA not available for embedding model, running on CPU.")
    embedding_model.eval()
    logging.info("Embedding model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading embedding model/tokenizer: {e}", exc_info=True)
    exit()

# ChromaDB 클라이언트 설정
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="processed_chunks")
    logging.info("ChromaDB client connected and collection retrieved.")
except Exception as e:
    logging.error(f"Error connecting to ChromaDB: {e}", exc_info=True)
    exit()

# --- OpenAI API 설정 ---
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set or found in .env file.")
    client = openai.OpenAI()
    logging.info("OpenAI API client configured.")
except Exception as e:
    logging.error(f"Error configuring OpenAI API: {e}", exc_info=True)
    logging.error("Please ensure the 'openai' library is installed and the OPENAI_API_KEY environment variable is set correctly in .env file.")
    exit()

# --- GPT 모델 선택 ---
GPT_MODEL_NAME = "gpt-3.5-turbo"
logging.info(f"Using GPT model: {GPT_MODEL_NAME}")

# --- Helper 함수 ---
def get_embedding(text):
    """텍스트 임베딩 생성 (BAAI/bge-m3 사용)"""
    try:
        inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        device = embedding_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = embedding_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.cpu().squeeze().tolist()
    except Exception as e:
        logging.error(f"Error generating embedding for text: '{text[:100]}...': {e}", exc_info=True)
        return None

def generate_gpt_response(context, max_tokens=1024):
    """OpenAI GPT API를 사용하여 응답 생성 (JSON 형식 추출 유도)"""
    logging.info(f"--- Function generate_gpt_response entered ---")
    if not context:
        logging.warning("Context is empty, cannot generate GPT response.")
        return None

    system_prompt = "You are a helpful assistant designed to extract specific information from Korean housing announcement documents and output it strictly as a JSON object."
    user_prompt = f"""다음은 주택 입주자 모집 공고문의 일부 내용입니다. 이 내용을 바탕으로 모집 일정 정보를 추출하여 **반드시 JSON 객체 형식**으로만 응답해주세요. 각 항목의 값이 명시적으로 언급되지 않았다면 null을 사용하세요. JSON 객체 외의 설명이나 다른 텍스트는 절대 포함하지 마세요.

요구 항목:
- "announcement_date": 모집공고일 (YYYY.MM.DD 형식)
- "online_application_period": {{
    "start": 인터넷 신청접수 시작일 (YYYY.MM.DD 형식),
    "end": 인터넷 신청접수 종료일 (YYYY.MM.DD 형식)
  }}
- "document_submission_period": {{
    "start": 서류제출 시작일 (YYYY.MM.DD 형식),
    "end": 서류제출 종료일 (YYYY.MM.DD 형식)
  }}
- "inspection_period": {{
    "start": 입주자격 검증 시작월 (YYYY.MM 형식),
    "end": 입주자격 검증 종료월 (YYYY.MM 형식)
  }}
- "winner_announcement": 당첨자 발표일시 (YYYY.MM.DD HH:mm 형식)
- "contract_period": {{
    "start": 주택공개 시작일 (YYYY.MM.DD 형식),
    "end": 주택공개 종료일 (YYYY.MM.DD 형식)
  }}
- "payment_period": {{
    "start": 계약체결 시작일 (YYYY.MM.DD 형식),
    "end": 계약체결 종료일 (YYYY.MM.DD 형식)
  }}
- "move_in_period": {{
    "start": 입주 시작일 (YYYY.MM.DD 형식),
    "end": 입주 종료일 (YYYY.MM.DD 형식)
  }}

[공고문 내용 시작]
{context}
[공고문 내용 끝]

JSON 응답:
"""

    logging.info("Sending request to OpenAI API...")
    start_time = time.time()
    response_content = None
    try:
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
        response_content = response.choices[0].message.content
        end_time = time.time()
        logging.info(f"OpenAI API response received in {end_time - start_time:.2f} seconds.")
        logging.debug(f"OpenAI API Raw Response Content:\n{response_content}")
        return response_content

    except Exception as e:
        logging.error(f"Error during OpenAI API call: {e}", exc_info=True)
        return None

def parse_gpt_json_output(gpt_response_content):
    """GPT API 응답 내용(JSON 문자열)을 파싱"""
    logging.info(f"--- Function parse_gpt_json_output entered ---")
    if not gpt_response_content:
        logging.warning("Input to parse_gpt_json_output is empty.")
        return None

    try:
        parsed_json = json.loads(gpt_response_content)
        logging.info("Successfully parsed JSON from GPT API response.")
        logging.debug(f"Parsed JSON: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}")
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from GPT API response: {e}", exc_info=True)
        logging.error(f"GPT API Response Content causing error:\n{gpt_response_content}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during JSON parsing: {e}", exc_info=True)
        return None

def extract_and_save_schedule_for_pdf(pdf_name, output_dir="extracted_schedule_gpt"):
    """특정 PDF의 모집 일정을 (의미 검색 + GPT API)로 추출하여 JSON 파일로 저장"""
    logging.info(f"--- Function extract_and_save_schedule_for_pdf entered for {pdf_name} ---")

    # 1. 의미 검색으로 관련 청크 가져오기
    search_queries = [
        "모집 일정",
        "신청 접수",
        "서류 제출",
        "당첨자 발표",
        "계약 체결",
        "입주 기간"
    ]
    n_results_to_fetch = 3  # 각 쿼리당 3개 결과만 가져오기

    logging.info(f"Performing semantic search for multiple queries within {pdf_name}...")
    context_text = ""
    
    try:
        for query in search_queries:
            logging.info(f"Processing query: '{query}'")
            query_embedding = get_embedding(query)
            if query_embedding is None:
                logging.error(f"Failed to generate embedding for query: '{query}'")
                continue

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results_to_fetch,
                where={"filename": pdf_name},
                include=["documents", "metadatas"]
            )

            if not results or not results.get('documents') or not results['documents'][0]:
                logging.warning(f"No results for query: '{query}'")
                continue

            retrieved_docs = results['documents'][0]
            retrieved_metas = results['metadatas'][0]

            chunks_with_meta = list(zip(retrieved_docs, retrieved_metas))
            chunks_with_meta.sort(key=lambda x: x[1].get('chunk_index', float('inf')))

            for doc, meta in chunks_with_meta:
                content_cleaned = re.sub(r'<h\d>.*?</h\d>\n?', '', doc, 1).strip()
                if content_cleaned and content_cleaned not in context_text:
                    context_text += content_cleaned + "\n\n"

        if not context_text:
            logging.warning(f"No relevant content found for any query in {pdf_name}")
            return False

        logging.info(f"Total combined context length: {len(context_text)} characters")
        logging.debug(f"First 1000 characters of context:\n{context_text[:1000]}")

    except Exception as e:
        logging.error(f"Error during semantic search for {pdf_name}: {e}", exc_info=True)
        return False

    # 2. GPT API 호출 및 파싱
    gpt_response_content = generate_gpt_response(context_text)
    if not gpt_response_content:
        logging.error(f"Failed to get response from GPT API for {pdf_name}")
        return False

    extracted_schedule = parse_gpt_json_output(gpt_response_content)
    if not extracted_schedule:
        logging.error(f"Failed to parse schedule using GPT API for {pdf_name}")
        return False

    # 3. 파일로 저장
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_schedule.json")

        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_schedule, f, ensure_ascii=False, indent=2)

        logging.info(f"Successfully saved schedule for {pdf_name} to {output_filename}")
        return True

    except Exception as e:
        logging.error(f"Error saving schedule JSON for {pdf_name}: {e}", exc_info=True)
        return False

# --- 메인 실행 부분 (모든 공고 처리) ---
if __name__ == "__main__":
    logging.info("Starting script to extract schedule information for all PDFs using GPT API...")

    # ChromaDB에서 모든 고유한 파일명 가져오기
    try:
        results = collection.get()
        unique_filenames = set(meta['filename'] for meta in results['metadatas'])
        logging.info(f"Found {len(unique_filenames)} unique PDF files in ChromaDB")
    except Exception as e:
        logging.error(f"Error retrieving filenames from ChromaDB: {e}", exc_info=True)
        exit()

    total_start_time = time.time()
    success_count = 0
    fail_count = 0

    # 각 PDF 파일에 대해 처리
    for pdf_name in unique_filenames:
        logging.info(f"Processing PDF: {pdf_name}")
        try:
            success = extract_and_save_schedule_for_pdf(pdf_name, output_dir="./extracted_schedule_gpt")
            if success:
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            logging.error(f"An unexpected error occurred during processing {pdf_name}: {e}", exc_info=True)
            fail_count += 1

    total_end_time = time.time()
    logging.info("=" * 50)
    logging.info(f"Finished processing all PDFs. Success: {success_count}, Failed: {fail_count}")
    logging.info(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")
    logging.info("=" * 50) 