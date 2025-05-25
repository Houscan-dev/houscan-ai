import os
import json
import datetime
import re
import chromadb
from chromadb.config import Settings
import time
from dotenv import load_dotenv # .env 파일 로드
import logging
import openai # OpenAI 라이브러리 임포트

# .env 파일 로드 (스크립트 시작 시점에 호출)
load_dotenv()

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# ChromaDB 클라이언트 설정
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="processed_chunks")
    logging.info("ChromaDB client connected and collection retrieved.")
except Exception as e:
    logging.error(f"Error connecting to ChromaDB: {e}", exc_info=True)
    exit()

# --- 모델 선택 ---
GPT_MODEL_NAME = "gpt-4-turbo"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
logging.info(f"Using GPT model: {GPT_MODEL_NAME}")
logging.info(f"Using Embedding model: {EMBEDDING_MODEL_NAME}")

# --- Helper 함수 ---

def get_embedding(text):
    """텍스트 임베딩 생성 (OpenAI text-embedding-3-large 사용)"""
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=text,
            encoding_format="float",
            dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding for text: '{text[:100]}...': {e}", exc_info=True)
        return None

def generate_gpt_response(context, max_tokens=4096):
    """ OpenAI GPT API를 사용하여 응답 생성 (서술형 텍스트 추출) """
    logging.info(f"--- Function generate_gpt_response entered ---")
    if not context:
        logging.warning("Context is empty, cannot generate GPT response.")
        return None

    system_prompt = "당신은 주택 입주자 모집 공고문을 분석하여 청년 신청자의 자격요건을 명확하고 일관된 형식으로 정리하는 전문가입니다. 반드시 '~습니다' 문체를 사용해주세요."
    user_prompt = f"""아래는 주택 입주자 모집 공고문에서 청년 신청자의 자격요건을 추출하여 정리하는 작업입니다. 
예시와 같은 형식과 스타일로 자격요건을 정리해주세요.

[예시 공고문]
주택공급신청자는 입주자모집공고일 현재 무주택자(본인)이며 미혼 상태여야 하며, 청년, 취업준비생, 대학생으로서 신청소득, 총자산, 자동차가액 기준을 충족해야 합니다. 무주택 요건 충족은 본인에 한하며 관련 규정에 따라 판단됩니다. 고졸자, 외국인, 재외국민은 신청이 불가합니다. 1인 1주택만 신청 가능하며, 중복 신청 시 전부 무효 처리됩니다. 입주자 모집공고일부터 계약 시까지 자격을 유지해야 하며, 당첨 후라도 자격요건 미달 시 계약이 취소될 수 있습니다. 부적격사유에 대한 소명 의무는 신청자에게 있습니다.

신청유형은 ① 대학생(서울특별시 소재 대학교 재학 중인 자), ② 취업준비생(졸업 또는 중퇴 후 미취업 상태로 취업준비 중인 자), ③ 청년(만 19세 이상 ~ 39세 이하) 중 하나를 선택해야 하며, 순위에 따라 자격이 나뉩니다.

1순위는 생계·의료·주거급여 수급자 가구, 한부모 가족, 차상위계층 가구이며, 2순위는 일반(도시근로자 월평균소득 100% 이하 및 자산 기준 충족)입니다. 3순위는 1, 2순위에 해당하지 않지만 소득 기준 충족자이며, 자산 기준은 완화되어 있습니다. 가구원 범위는 본인과 부모를 포함하고, 3순위 단독세대주는 본인만 포함됩니다.

자산 기준은 2순위의 경우 총자산 2억 8,800만 원 이하, 자동차 2,468만 원 이하이며, 3순위는 총자산 2억 3,700만 원 이하, 자동차 2,468만 원 이하입니다. 소득 기준은 1인 가구 기준 50% 이하 1,322,574원, 100% 이하 2,645,147원 등으로 가구원 수에 따라 상이합니다.

[예시 응답]
주택공급신청자는 입주자모집공고일 현재 무주택자(본인)이며 미혼 상태여야 하며, 청년, 취업준비생, 대학생으로서 신청소득, 총자산, 자동차가액 기준을 충족해야 합니다. 신청유형은 ① 대학생(서울특별시 소재 대학교 재학 중인 자), ② 취업준비생(졸업 또는 중퇴 후 미취업 상태로 취업준비 중인 자), ③ 청년(만 19세 이상 ~ 39세 이하) 중 하나를 선택해야 하며, 순위에 따라 자격이 나뉩니다. 1순위는 생계·의료·주거급여 수급자 가구, 한부모 가족, 차상위계층 가구이며, 2순위는 일반(도시근로자 월평균소득 100% 이하 및 자산 기준 충족)입니다. 3순위는 1, 2순위에 해당하지 않지만 소득 기준 충족자이며, 자산 기준은 완화되어 있습니다. 가구원 범위는 본인과 부모를 포함하고, 3순위 단독세대주는 본인만 포함됩니다. 자산 기준은 2순위의 경우 총자산 2억 8,800만 원 이하, 자동차 2,468만 원 이하이며, 3순위는 총자산 2억 3,700만 원 이하, 자동차 2,468만 원 이하입니다. 소득 기준은 1인 가구 기준 50% 이하 1,322,574원, 100% 이하 2,645,147원 등으로 가구원 수에 따라 상이합니다. 무주택 요건 충족은 본인에 한하며 관련 규정에 따라 판단됩니다. 고졸자, 외국인, 재외국민은 신청이 불가합니다. 1인 1주택만 신청 가능하며, 중복 신청 시 전부 무효 처리됩니다.

[새로운 공고문]
{context}

위 공고문의 자격요건을 예시 응답과 같은 형식과 문체로 작성해주세요."""

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
            max_tokens=max_tokens
        )
        response_content = response.choices[0].message.content
        end_time = time.time()
        logging.info(f"OpenAI API response received in {end_time - start_time:.2f} seconds.")
        logging.debug(f"OpenAI API Raw Response Content:\n{response_content}")
        return response_content

    except openai.APIError as e:
        logging.error(f"OpenAI API returned an API Error: {e}", exc_info=True)
    except openai.APIConnectionError as e:
        logging.error(f"Failed to connect to OpenAI API: {e}", exc_info=True)
    except openai.RateLimitError as e:
        logging.error(f"OpenAI API request exceeded rate limit: {e}", exc_info=True)
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI API key error: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred during OpenAI API call: {e}", exc_info=True)

    return None

def parse_gpt_json_output(gpt_response_content):
    """ GPT API 응답 내용(JSON 문자열)을 파싱 """
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


# --- 메인 처리 함수 ---

def extract_and_save_criteria_for_pdf(pdf_name, output_dir="criteria3"):
    """
    특정 PDF의 자격 요건을 (의미 검색 + GPT API)로 추출하여 텍스트 파일로 저장
    """
    logging.info(f"--- Function extract_and_save_criteria_for_pdf entered for {pdf_name} ---")

    # 1. 의미 검색으로 관련 청크 가져오기
    search_queries = [
        "신청자격",
        "자격요건",
        "모집자격"
    ]
    n_results_to_fetch = 10  # 각 쿼리당 10개 결과만 가져오기

    logging.info(f"Performing semantic search for multiple queries within {pdf_name}...")
    context_text = ""  # 컨텍스트 초기화
    
    try:
        # 각 쿼리에 대해 검색 수행
        for query in search_queries:
            logging.info(f"Processing query: '{query}'")
            query_embedding = get_embedding(query)
            if query_embedding is None:
                logging.error(f"Failed to generate embedding for query: '{query}'")
                continue

            # collection.query 사용 (의미 기반 검색)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results_to_fetch,
                where={"filename": pdf_name},
                include=["documents", "metadatas"]
            )

            if not results or not results.get('documents') or not results['documents'][0]:
                logging.warning(f"No results for query: '{query}'")
                continue

            # 결과 처리
            retrieved_docs = results['documents'][0]
            retrieved_metas = results['metadatas'][0]

            # 청크 정렬 및 정리
            chunks_with_meta = list(zip(retrieved_docs, retrieved_metas))
            chunks_with_meta.sort(key=lambda x: x[1].get('chunk_index', float('inf')))

            # HTML 태그 제거 및 내용 정리
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

    # 2. GPT API 호출
    gpt_response = generate_gpt_response(context_text)
    if not gpt_response:
        logging.error(f"Failed to get response from GPT API for {pdf_name}")
        return False

    # 3. 파일로 저장
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.splitext(pdf_name)[0]
        safe_base_filename = re.sub(r'[^\w\-]+', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_criteria.json")

        # 텍스트 형식으로 저장
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(gpt_response)

        logging.info(f"Successfully saved criteria for {pdf_name} to {output_filename}")
        return True

    except Exception as e:
        logging.error(f"Error saving criteria for {pdf_name}: {e}", exc_info=True)
        return False

# --- 메인 실행 부분 (모든 공고 처리) ---
if __name__ == "__main__":
    logging.info("Starting script to extract eligibility criteria for all PDFs using GPT API...")

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
            success = extract_and_save_criteria_for_pdf(pdf_name, output_dir="criteria3")
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