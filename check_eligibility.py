import os
import torch
import json
import datetime
import re
import chromadb
from chromadb.config import Settings
# LLM 사용을 위한 transformers 추가
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel # 또는 AutoModelForSeq2SeqLM
import time # LLM 응답 시간 측정을 위해 추가 (선택 사항)

# --- 기존 ChromaDB 및 임베딩 설정 (유지) ---
# 임베딩 모델 로드 (기존 코드와 동일하게 사용한다고 가정)
embedding_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
embedding_model = AutoModel.from_pretrained("BAAI/bge-m3")
embedding_model.eval()

# ChromaDB 클라이언트 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="processed_chunks")

def get_embedding(text):
    """텍스트 임베딩 생성 (기존 코드 활용)"""
    inputs = embedding_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().tolist()

# --- LLM 로딩 및 설정 ---
# 허깅페이스에서 사용할 LLM 모델 선택 (예시, 필요시 변경)
# 한국어 지원 및 Instruction Following 능력이 좋은 모델 선택 필요
# 예시 1: KoAlpaca (Polyglot 기반) - 비교적 가벼움
# llm_model_name = "beomi/KoAlpaca-Polyglot-12.8B"
# 예시 2: Smaller T5 variant (실험 필요)
# llm_model_name = "google/flan-t5-large" # 한국어 성능 및 JSON 출력 능력 확인 필요
# 예시 3: Larger model (리소스 요구량 높음)
# llm_model_name = "HuggingFaceH4/zephyr-7b-beta" # 한국어 성능 및 JSON 출력 능력 확인 필요

# 실제 사용 가능한 모델로 교체해야 합니다. 여기서는 예시 이름을 사용합니다.
# 리소스 제약이 있다면 더 작은 모델을 사용하거나 API 형태의 LLM 사용을 고려해야 합니다.
llm_model_name = "Upstage/SOLAR-10.7B-Instruct-v1.0" # <--- !!! 중요: 실제 사용 가능한 LLM 모델 이름으로 반드시 변경하세요 !!!
                        # gpt2는 예시일 뿐이며, 한국어/Instruction/JSON 출력에 적합하지 않을 수 있습니다.
                        # KoAlpaca, Solar, Llama-ko 등 고려

print(f"Loading LLM model: {llm_model_name}...")
try:
    # 모델 로딩 방식은 모델 타입(CausalLM, Seq2SeqLM)에 따라 다를 수 있음
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    # CausalLM 예시 (GPT, Llama 등)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
    # Seq2SeqLM 예시 (T5, BART 등)
    # llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

    # GPU 사용 설정 (가능하다면)
    if torch.cuda.is_available():
        llm_model.to("cuda")
    llm_model.eval() # 추론 모드 설정

    # 파이프라인 사용 (더 간편할 수 있음)
    # text_generator = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device=0 if torch.cuda.is_available() else -1)
    print("LLM model loaded successfully.")
except Exception as e:
    print(f"Error loading LLM model: {e}")
    print("Please ensure the model name is correct and you have enough resources.")
    # LLM 로딩 실패 시 종료 또는 다른 처리
    exit()

# --- 자격 조건 추출 및 검증 함수 ---

def get_relevant_chunks(pdf_name, relevant_titles):
    """ 특정 PDF에서 관련 제목을 가진 청크들을 가져옴 """
    where_conditions = {
        "filename": pdf_name,
        "$or": [{"title": title} for title in relevant_titles]
    }
    results = collection.get(where=where_conditions)

    if not results or not results['documents']:
        print(f"Warning: No chunks found for titles {relevant_titles} in {pdf_name}")
        return ""

    # 청크 인덱스 순서대로 정렬 (선택 사항, 문맥 연결에 도움될 수 있음)
    chunks_with_meta = list(zip(results['documents'], results['metadatas']))
    chunks_with_meta.sort(key=lambda x: x[1].get('chunk_index', 0))

    # 관련 청크 내용 합치기
    # HTML 태그 제거 (LLM 입력 전처리)
    combined_text = ""
    for doc, meta in chunks_with_meta:
        # 제목 태그 제거 및 기본 전처리
        content_cleaned = re.sub(r'<h\d>.*?</h\d>\n?', '', doc, 1)
        content_cleaned = content_cleaned.strip()
        combined_text += content_cleaned + "\n\n" # 청크 간 구분

    return combined_text

def generate_llm_response(context, max_new_tokens=512):
    """ LLM을 사용하여 응답 생성 (JSON 형식 추출 유도) """

    # --- 프롬프트 정의 ---
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

    print("\n--- Sending Prompt to LLM ---")
    # print(prompt) # 프롬프트 내용 확인 (디버깅 시)
    print("--- End of Prompt ---")

    start_time = time.time()
    try:
        # 방법 1: pipeline 사용 시
        # generated = text_generator(prompt, max_length=len(prompt.split()) + max_new_tokens, num_return_sequences=1, pad_token_id=llm_tokenizer.eos_token_id)
        # response_text = generated[0]['generated_text'][len(prompt):] # 프롬프트 부분 제외

        # 방법 2: 모델 직접 사용 시 (CausalLM 예시)
        inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048) # 모델의 최대 컨텍스트 길이에 맞춰 조정
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")

        # 토큰 생성 옵션 설정
        # temperature, top_p 등 조절하여 결과 품질 개선 시도 가능
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3, # 반복 줄이기
            # early_stopping=True, # 필요시
            pad_token_id=llm_tokenizer.eos_token_id,
            # temperature=0.7, # 약간의 창의성 허용
            # top_p=0.9,
            # do_sample=True # 샘플링 사용 여부
        )
        response_text = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        end_time = time.time()
        print(f"LLM response generated in {end_time - start_time:.2f} seconds.")
        print("\n--- LLM Raw Response ---")
        print(response_text)
        print("--- End of LLM Raw Response ---")
        return response_text

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        return None

def parse_llm_json_output(llm_output):
    """ LLM 출력에서 JSON 부분만 추출하고 파싱 """
    if not llm_output:
        return None

    # LLM 응답에서 JSON 객체만 정확히 추출 시도
    # 가장 간단한 방법: ```json ... ``` 블록 찾기
    match_code_block = re.search(r'```json\s*(\{.*?\})\s*```', llm_output, re.DOTALL | re.IGNORECASE)
    # 또는 응답 시작이 { 로 시작하는지 확인
    match_direct_json = re.search(r'^\s*(\{.*?\})\s*$', llm_output, re.DOTALL)

    json_str = None
    if match_code_block:
        json_str = match_code_block.group(1)
    elif match_direct_json:
         json_str = match_direct_json.group(1)
    else:
        # LLM이 JSON만 반환하도록 유도했으므로, 전체 응답이 JSON일 수 있음
        # 또는, JSON이 깨졌거나 다른 텍스트와 섞였을 수 있음
        # 시작 '{' 와 마지막 '}'를 찾아 추출 시도 (가장 불안정)
        start_index = llm_output.find('{')
        end_index = llm_output.rfind('}')
        if start_index != -1 and end_index != -1 and start_index < end_index:
            json_str = llm_output[start_index : end_index + 1]
        else:
            print("Warning: Could not find JSON structure in LLM output.")
            return None

    try:
        parsed_json = json.loads(json_str)
        print("\n--- Parsed JSON from LLM ---")
        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        print("--- End of Parsed JSON ---")
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM output: {e}")
        print("LLM Output causing error:\n", json_str)
        return None

# 금액 단위 변환 함수 (기존 코드 활용)
def convert_to_won(value_str, unit_indicator):
    """ 금액 단위 변환 (LLM 추출값은 이미 숫자일 수 있으므로 약간 수정) """
    try:
        # LLM이 이미 숫자로 추출했을 경우
        if isinstance(value_str, (int, float)):
             return int(value_str)
        # LLM이 문자열로 추출했을 경우 (예: "3억 803만원") - 추가 처리 필요
        # 간단히 숫자만 추출 시도
        value_str = re.sub(r'[^\d]', '', value_str)
        value = int(value_str)

        # 원본 텍스트에 '억원'이나 '만원'이 있었는지 확인 필요 (현재 구조로는 어려움)
        # 여기서는 LLM이 원 단위로 잘 추출했다고 가정하거나,
        # 프롬프트에서 명시적으로 원 단위로 반환하도록 요구해야 함.
        # 만약 LLM이 "33700만원" 처럼 반환하면 후처리 필요.
        # -> 가장 좋은 방법은 프롬프트에서 '원 단위 정수'로 명시하는 것.
        return value
    except Exception as e:
        print(f"Warning: Could not convert value '{value_str}' to Won: {e}")
        return None


def check_eligibility_with_llm(pdf_name, user_info):
    """ LLM을 사용하여 특정 공고문의 자격 요건 검증 """

    # 1. 관련 청크 검색 (제목 기반)
    # 이 공고문의 실제 제목 태그를 확인하여 수정 필요
    relevant_titles = [
        "1.  공통사항",  # chunk_index: 4
        "□ 입주자모집공고일(2025.03.07) 현재 무주택세대구성원으로서 아래의 요건을 모두 갖춘 자", # chunk_index: 8
        "□ 소득 및 자산 기준" # chunk_index: 9, 10, 11
        # 필요시 다른 관련 제목 추가
    ]
    context_text = get_relevant_chunks(pdf_name, relevant_titles)

    if not context_text:
        print(f"Error: Could not retrieve relevant context for {pdf_name}")
        return {"pdf_name": pdf_name, "eligible": None, "reason": "Relevant context not found"}

    # 2. LLM 호출하여 자격 조건 추출
    llm_output = generate_llm_response(context_text)
    extracted_criteria = parse_llm_json_output(llm_output)

    if not extracted_criteria:
        print(f"Error: Failed to extract criteria using LLM for {pdf_name}")
        return {"pdf_name": pdf_name, "eligible": None, "reason": "Failed to extract criteria using LLM"}

    # 3. 자격 조건 비교 분석
    analysis_result = {
        "pdf_name": pdf_name,
        "eligible": True, # 기본값을 True로 설정하고, 미충족 시 False로 변경
        "matched_conditions": [],
        "unmet_conditions": [],
        "extracted_criteria": extracted_criteria # LLM이 추출한 원본 정보 포함
    }

    # 나이 검증
    birth_year = int(user_info["birth_date"][:4])
    current_year = datetime.datetime.now().year
    # 만 나이 계산 (한국식 나이 대신)
    # 생일이 지났는지 여부까지 고려하면 더 정확하지만, 여기서는 연도 기준으로 계산
    age_international = current_year - birth_year
    # 만약 공고일 기준 만 나이가 필요하다면 공고일 사용 (여기서는 현재 연도 기준)
    age_criterion_met = True
    if extracted_criteria.get("age_min") is not None and age_international < extracted_criteria["age_min"]:
        age_criterion_met = False
    if extracted_criteria.get("age_max") is not None and age_international > extracted_criteria["age_max"]:
        age_criterion_met = False

    if age_criterion_met:
         if extracted_criteria.get("age_min") is not None or extracted_criteria.get("age_max") is not None:
              age_req_str = f"만 {extracted_criteria.get('age_min', '?')}세 ~ {extracted_criteria.get('age_max', '?')}세"
              analysis_result["matched_conditions"].append(f"연령 조건 충족: {age_req_str} (현재 만 {age_international}세)")
    else:
        age_req_str = f"만 {extracted_criteria.get('age_min', '?')}세 ~ {extracted_criteria.get('age_max', '?')}세"
        analysis_result["eligible"] = False
        analysis_result["unmet_conditions"].append(f"연령 조건 미충족: 요구사항={age_req_str}, 사용자=만 {age_international}세")

    # 혼인 상태 검증
    must_be_unmarried = extracted_criteria.get("must_be_unmarried")
    if must_be_unmarried is True: # LLM이 boolean으로 반환했다고 가정
        # user_info에 혼인 상태 필드가 필요함. 없으므로 임시로 '미혼'으로 가정.
        user_is_married = False # 예시: 실제 user_info에서 가져와야 함
        if user_is_married:
            analysis_result["eligible"] = False
            analysis_result["unmet_conditions"].append("혼인 조건 미충족: 미혼이어야 함")
        else:
            analysis_result["matched_conditions"].append("혼인 조건 충족: 미혼")
    elif must_be_unmarried is False:
         analysis_result["matched_conditions"].append("혼인 조건 무관")
    # null인 경우는 조건이 없거나 추출 실패로 간주

    # 소득 검증
    income_limit_percent = extracted_criteria.get("income_limit_percent")
    if income_limit_percent is not None:
        user_income_percent_str = user_info["income_range"].replace("% 이하", "").strip()
        try:
            user_income_percent = int(user_income_percent_str)
            if user_income_percent > income_limit_percent:
                analysis_result["eligible"] = False
                analysis_result["unmet_conditions"].append(f"소득 기준 미충족: 요구사항={income_limit_percent}% 이하, 사용자={user_income_percent}%")
            else:
                analysis_result["matched_conditions"].append(f"소득 기준 충족: {income_limit_percent}% 이하 (사용자 {user_income_percent}%)")
        except ValueError:
             analysis_result["unmet_conditions"].append(f"소득 기준 검증 불가: 사용자 소득 정보 오류 ('{user_info['income_range']}')")


    # 총자산 검증
    total_asset_limit = extracted_criteria.get("total_asset_limit_won")
    if total_asset_limit is not None:
        user_total_assets = int(user_info["total_assets"])
        if user_total_assets > total_asset_limit:
            analysis_result["eligible"] = False
            analysis_result["unmet_conditions"].append(f"총자산 기준 미충족: 요구사항={total_asset_limit:,.0f}원 이하, 사용자={user_total_assets:,.0f}원")
        else:
            analysis_result["matched_conditions"].append(f"총자산 기준 충족: {total_asset_limit:,.0f}원 이하")

    # 자동차 자산 검증
    car_asset_limit = extracted_criteria.get("car_asset_limit_won")
    if car_asset_limit is not None:
        user_car_value = int(user_info["car_value"])
        # LLM이 차량 가액 기준을 어떻게 해석했는지 중요 (0원 초과 불가 vs 특정 금액 이하)
        # 예시 공고문은 '3,803만원 이하' 이므로 이하로 비교
        if user_car_value > car_asset_limit:
             analysis_result["eligible"] = False
             analysis_result["unmet_conditions"].append(f"자동차 기준 미충족: 요구사항={car_asset_limit:,.0f}원 이하, 사용자={user_car_value:,.0f}원")
        else:
             analysis_result["matched_conditions"].append(f"자동차 기준 충족: {car_asset_limit:,.0f}원 이하")
    # 만약 LLM이 차량 보유 자체를 금지하는 것으로 판단했다면 (예: car_allowed=False), 로직 변경 필요

    # 무주택 요건 검증
    must_be_homeless = extracted_criteria.get("must_be_homeless")
    if must_be_homeless is True:
         # user_info에 무주택 여부 필드가 필요함. 없으므로 임시로 '무주택'으로 가정.
         user_is_homeless = True # 예시: 실제 user_info에서 가져와야 함
         if not user_is_homeless:
              analysis_result["eligible"] = False
              analysis_result["unmet_conditions"].append("무주택 조건 미충족: 무주택 세대구성원이어야 함")
         else:
              analysis_result["matched_conditions"].append("무주택 조건 충족")
    # False나 null인 경우는 조건이 없거나, 유주택자도 가능하거나, 추출 실패로 간주

    # 복잡한 논리 조건 (예: AND, OR)은 현재 구조에서 처리하기 어려움
    # LLM 프롬프트에서 더 복잡한 구조를 요청하거나, 후처리 로직 강화 필요

    return analysis_result

def print_eligibility_result_llm(result):
    """ LLM 기반 자격 검증 결과 출력 """
    print("\n" + "=" * 80)
    print(f"📋 공고문: {result['pdf_name']}")
    print("=" * 80)

    if result.get("eligible") is None:
        print(f"\n⚠️ 자격 검증 중 오류 발생: {result.get('reason', 'Unknown error')}")
    elif result["eligible"]:
        print("\n✅ 지원 가능합니다!")
    else:
        print("\n❌ 지원 자격이 충족되지 않습니다.")

    print("\n[LLM 추출 조건 (참고)]")
    if result.get("extracted_criteria"):
        print(json.dumps(result["extracted_criteria"], indent=2, ensure_ascii=False))
    else:
        print("추출된 조건 정보 없음")


    print("\n[충족된 조건]")
    if result.get("matched_conditions"):
        for condition in result["matched_conditions"]:
            print(f"✓ {condition}")
    else:
        print("충족된 조건 없음")

    if result.get("unmet_conditions"):
        print("\n[미충족된 조건]")
        for condition in result["unmet_conditions"]:
            print(f"✗ {condition}")
    else:
         # 자격 미달인데 미충족 조건이 없는 경우 (오류 가능성)
         if result.get("eligible") == False:
              print("미충족된 특정 조건이 식별되지 않았으나, 자격 요건 미달입니다.")


    print("=" * 80)

# --- 메인 실행 부분 ---
if __name__ == "__main__":
    # 사용자 정보 (예시)
    user_info = {
        "birth_date": "19990101", # 만 나이 계산 위해 YYYYMMDD 형식 권장
        "gender": "남성",
        "university_status": "재학 중", # LLM 프롬프트에 추가하여 조건 추출 요청 가능
        "recent_graduate": "예",     # "
        "employed": "아니오",        # "
        "job_seeking": "아니오",       # "
        "household_type": "생계·의료·주거급여 수급자 가구", # "
        "parents_own_house": "아니요", # "
        "disability_in_family": "예", # "
        "application_count": 2,
        "total_assets": 100000000, # 1억원 (테스트 용도)
        "car_value": 5000000,      # 5백만원 (테스트 용도)
        "income_range": "100% 이하" # 소득 분위 또는 비율 필요
        # 추가 필드: marital_status (Married/Unmarried), is_homeless (True/False) 등 필요
    }

    # 검증할 공고문 파일명
    # pdf_name = "[마을과집]SH특화형 매입임대주택(청년) 입주자 모집 공고문_20250307.pdf"
    pdf_name = "[마을과집]SH특화형 매입임대주택(청년) 입주자 모집 공고문_20250307.pdf" # 파일 확장자 제거 필요

    # 자격 검증 실행
    print(f"\n🚀 '{pdf_name}' 공고문에 대한 자격 검증 시작 (LLM 기반)")
    eligibility_result = check_eligibility_with_llm(pdf_name, user_info)

    # 결과 출력
    print_eligibility_result_llm(eligibility_result)