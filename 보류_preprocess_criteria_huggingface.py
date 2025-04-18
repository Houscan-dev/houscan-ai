import os
import torch
import json
import datetime
import re
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel # 또는 AutoModelForSeq2SeqLM
import time
import logging # 로깅 추가

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 기존 ChromaDB 및 임베딩 설정 (유지) ---
try:
    embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    embedding_model = AutoModel.from_pretrained("BAAI/bge-m3")
    embedding_model.eval()
    logging.info("Embedding model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading embedding model: {e}", exc_info=True)
    exit()

# ChromaDB 클라이언트 설정
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="processed_chunks")
    logging.info("ChromaDB client connected and collection retrieved.")
except Exception as e:
    logging.error(f"Error connecting to ChromaDB: {e}", exc_info=True)
    exit()

# --- LLM 로딩 및 설정 ---
llm_model_name = "Upstage/SOLAR-10.7B-Instruct-v1.0" # <--- 실제 사용 가능한 모델로 변경하세요!
logging.info(f"Loading LLM model: {llm_model_name}...")
llm_tokenizer = None
llm_model = None
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        # torch_dtype=torch.float16, # VRAM 절약을 위해 float16 사용 고려 (호환성 확인 필요)
        # device_map='auto' # 여러 GPU에 분산 로딩 (VRAM 부족 시)
    )

    if torch.cuda.is_available():
        logging.info("CUDA available, moving LLM model to GPU.")
        # device_map='auto'를 사용하지 않는 경우 수동으로 GPU 할당
        # llm_model.to("cuda")
    else:
        logging.warning("CUDA not available, LLM model will run on CPU (might be very slow).")

    llm_model.eval()
    logging.info("LLM model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading LLM model: {e}", exc_info=True)
    logging.error("Please ensure the model name is correct and you have enough resources (RAM/VRAM).")
    exit()

# --- 자격 조건 추출 및 저장 관련 함수 ---

def get_relevant_chunks(pdf_name, relevant_titles):
    """ 특정 PDF에서 관련 제목을 가진 청크들을 가져옴 """
    try:
        where_conditions = {
            "filename": pdf_name,
            "$or": [{"title": title} for title in relevant_titles]
        }
        # `include` 파라미터를 사용하여 필요한 데이터만 가져오도록 최적화
        results = collection.get(where=where_conditions, include=["documents", "metadatas"])

        if not results or not results.get('documents'):
            logging.warning(f"No relevant chunks found for titles {relevant_titles} in {pdf_name}")
            return ""

        # 청크 인덱스 순서대로 정렬
        chunks_with_meta = list(zip(results['documents'], results['metadatas']))
        chunks_with_meta.sort(key=lambda x: x[1].get('chunk_index', 0))

        combined_text = ""
        for doc, meta in chunks_with_meta:
            content_cleaned = re.sub(r'<h\d>.*?</h\d>\n?', '', doc, 1).strip()
            combined_text += content_cleaned + "\n\n" # 청크 간 구분

        logging.info(f"Retrieved and combined {len(chunks_with_meta)} relevant chunks for {pdf_name}.")
        return combined_text.strip()
    except Exception as e:
        logging.error(f"Error retrieving chunks for {pdf_name}: {e}", exc_info=True)
        return ""

def generate_llm_response(context, max_new_tokens=512):
    """ LLM을 사용하여 응답 생성 (JSON 형식 추출 유도) """
    if not llm_model or not llm_tokenizer:
        logging.error("LLM model or tokenizer not loaded.")
        return None

    prompt = f"""다음은 주택 입주자 모집 공고문의 일부 내용입니다. 이 내용을 바탕으로 청년 신청자의 주요 입주 자격 요건을 추출하여 **반드시 JSON 형식**으로만 응답해주세요. 각 항목의 값이 명시적으로 언급되지 않았다면 null을 사용하세요. JSON 외의 설명은 절대 포함하지 마세요.

요구 항목:
- "age_min": 최소 나이 (만 나이 기준, 정수)
- "age_max": 최대 나이 (만 나이 기준, 정수)
- "must_be_unmarried": 혼인 상태 (미혼이어야 하는지 여부, 불리언)
- "income_limit_percent": 소득 기준 (전년도 도시근로자 월평균소득 대비 비율, 정수, 예: 100)
- "total_asset_limit_won": 총자산 기준 (세대 기준, 원 단위, 정수)
- "car_asset_limit_won": 자동차 기준 (세대 기준, 원 단위, 정수)
- "must_be_homeless": 무주택 요건 (세대 기준, 불리언)

[공고문 내용 시작]
{context}
[공고문 내용 끝]

JSON 응답:
"""

    logging.info("Sending prompt to LLM...")
    start_time = time.time()
    response_text = ""
    try:
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500) # 모델 컨텍스트 길이 고려 조정

        # 모델이 로드된 디바이스로 텐서 이동
        device = llm_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad(): # 추론 시 그래디언트 계산 비활성화
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=3,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id, # 명시적으로 종료 토큰 설정
                temperature=0.1, # 일관성 있는 출력을 위해 낮은 온도 설정
                top_p=0.9,
                do_sample=True # 약간의 샘플링 사용 (False로 하면 더 결정적)
            )
        # 생성된 부분만 디코딩 (입력 프롬프트 제외)
        response_text = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        end_time = time.time()
        logging.info(f"LLM response generated in {end_time - start_time:.2f} seconds.")
        logging.debug(f"LLM Raw Response:\n{response_text}") # 디버그 레벨에서만 로우 응답 기록
        return response_text

    except Exception as e:
        logging.error(f"Error during LLM generation: {e}", exc_info=True)
        logging.error(f"Prompt length: {len(prompt)}")
        logging.error(f"Input tensor shape: {inputs['input_ids'].shape if 'inputs' in locals() else 'N/A'}")
        return None

def parse_llm_json_output(llm_output):
    """ LLM 출력에서 JSON 부분만 추출하고 파싱 """
    if not llm_output:
        return None

    # JSON 객체 추출 시도 (더 견고하게)
    json_str = None
    # 1. ```json ... ``` 블록 우선 탐색
    match_code_block = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL | re.IGNORECASE)
    if match_code_block:
        json_str = match_code_block.group(1)
    else:
        # 2. 응답이 { 로 시작하고 } 로 끝나는 경우
        match_direct_json = re.search(r'^\s*(\{.*?\})\s*$', llm_output, re.DOTALL)
        if match_direct_json:
            json_str = match_direct_json.group(1)
        else:
            # 3. 응답 내에서 가장 그럴듯한 { ... } 블록 찾기 (최후의 수단)
            start_index = llm_output.find('{')
            end_index = llm_output.rfind('}')
            if start_index != -1 and end_index != -1 and start_index < end_index:
                potential_json = llm_output[start_index : end_index + 1]
                # 간단한 유효성 검사 (중첩된 괄호 수 등) - 완벽하진 않음
                if potential_json.count('{') == potential_json.count('}'):
                     json_str = potential_json
            if not json_str:
                 logging.warning("Could not find JSON structure in LLM output.")
                 logging.warning(f"Problematic LLM output: {llm_output[:500]}...") # 앞부분 일부 로깅
                 return None

    try:
        parsed_json = json.loads(json_str)
        logging.info("Successfully parsed JSON from LLM output.")
        logging.debug(f"Parsed JSON: {json.dumps(parsed_json, indent=2, ensure_ascii=False)}")
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON from LLM output: {e}", exc_info=True)
        logging.error(f"LLM Output (or extracted part) causing error:\n{json_str}")
        return None

def get_all_pdf_filenames():
    """ ChromaDB에서 저장된 모든 고유 PDF 파일명 리스트를 가져옴 """
    try:
        # distinct=True 와 유사한 효과를 내기 위해 metadata만 가져와 처리
        results = collection.get(include=['metadatas'])
        if results and results['metadatas']:
            pdf_files = sorted(list(set(meta['filename'] for meta in results['metadatas'] if 'filename' in meta)))
            logging.info(f"Found {len(pdf_files)} unique PDF filenames in ChromaDB.")
            return pdf_files
        else:
            logging.warning("No documents found in ChromaDB collection.")
            return []
    except Exception as e:
        logging.error(f"Error retrieving PDF list from ChromaDB: {e}", exc_info=True)
        return []

def extract_and_save_criteria_for_pdf(pdf_name, output_dir="extracted_criteria"):
    """ 특정 PDF의 자격 요건을 LLM으로 추출하여 JSON 파일로 저장 """
    logging.info(f"Processing: {pdf_name}")

    # 1. 관련 청크 검색
    # 이 제목들은 예시이며, 실제 데이터에 맞게 조정 필요
    relevant_titles = [
        "1.  공통사항",
        "□ 입주자모집공고일(2025.03.07) 현재 무주택세대구성원으로서 아래의 요건을 모두 갖춘 자",
        "□ 소득 및 자산 기준",
        "Ⅲ. 입주대상자 자격요건", # 일반적인 제목 추가
        "입주자격",
        "신청자격"
    ]
    # 중복 제거
    relevant_titles = sorted(list(set(relevant_titles)))

    context_text = get_relevant_chunks(pdf_name, relevant_titles)

    if not context_text:
        logging.warning(f"Skipping {pdf_name} due to missing relevant context.")
        return False # 처리 실패

    # 2. LLM 호출 및 파싱
    llm_output = generate_llm_response(context_text)
    extracted_criteria = parse_llm_json_output(llm_output)

    if not extracted_criteria:
        logging.error(f"Failed to extract or parse criteria for {pdf_name}.")
        return False # 처리 실패

    # 3. 파일로 저장
    try:
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 출력 파일명 생성 (원본 PDF명에서 확장자 제거 + 접미사 추가)
        base_filename = os.path.splitext(pdf_name)[0]
        # 파일명으로 부적합한 문자 제거/교체 (선택 사항)
        safe_base_filename = re.sub(r'[\\/*?:"<>|]', '_', base_filename)
        output_filename = os.path.join(output_dir, f"{safe_base_filename}_criteria.json")

        # JSON 파일 저장
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_criteria, f, ensure_ascii=False, indent=2)

        logging.info(f"Successfully saved criteria for {pdf_name} to {output_filename}")
        return True # 처리 성공

    except Exception as e:
        logging.error(f"Error saving criteria JSON for {pdf_name}: {e}", exc_info=True)
        return False # 처리 실패

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    logging.info("Starting pre-processing script to extract eligibility criteria...")

    # 처리할 PDF 목록 가져오기
    all_pdfs = get_all_pdf_filenames()

    if not all_pdfs:
        logging.warning("No PDF files found to process. Exiting.")
        exit()

    processed_count = 0
    failed_count = 0
    total_start_time = time.time()

    # 각 PDF 파일에 대해 자격 조건 추출 및 저장 실행
    for pdf_file in all_pdfs:
        pdf_start_time = time.time()
        success = extract_and_save_criteria_for_pdf(pdf_file, output_dir="./extracted_criteria")
        pdf_end_time = time.time()
        if success:
            processed_count += 1
            logging.info(f"Finished processing {pdf_file} in {pdf_end_time - pdf_start_time:.2f} seconds.")
        else:
            failed_count += 1
            logging.error(f"Failed to process {pdf_file} after {pdf_end_time - pdf_start_time:.2f} seconds.")
        # 각 파일 처리 후 잠시 대기 (선택 사항, API 호출 제한 등 고려)
        # time.sleep(1)

    total_end_time = time.time()
    logging.info("=" * 50)
    logging.info("Pre-processing finished.")
    logging.info(f"Total PDFs processed: {processed_count}")
    logging.info(f"Total PDFs failed: {failed_count}")
    logging.info(f"Total execution time: {total_end_time - total_start_time:.2f} seconds.")
    logging.info("Extracted criteria saved in './extracted_criteria' directory.")
    logging.info("=" * 50)