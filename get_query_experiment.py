import os
import torch
import json
import csv
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import datetime
import re

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")
model.eval()

# ChromaDB 클라이언트 설정
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="processed_chunks")

def get_embedding(text):
    """텍스트 임베딩 생성"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().tolist()

def search_chunks(query_text, top_k=3, pdf_name=None):
    """청크 검색 함수"""
    query_embedding = get_embedding("Represent this sentence for retrieval: " + query_text)
    
    # PDF 파일명이 지정된 경우 해당 파일 내에서만 검색
    if pdf_name:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"filename": pdf_name}
        )
    else:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

    print(f"\n🔍 '{query_text}'에 대한 검색 결과 (Top {top_k}):")
    if results and results['ids'] and results['documents'] and results['metadatas']:
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            content = results['documents'][0][i]
            
            print(f"\n[{i+1}] 문서: {metadata['filename']}")
            print(f"섹션: {metadata.get('title', '제목 없음')}")
            
            # 내용 출력 (제목 태그 제거)
            content_without_title = re.sub(r'<h\d>.*?</h\d>\n?', '', content, 1)
            print(f"내용: {content_without_title[:200]}...")  # 처음 200자만 출력
            print("-" * 80)
    else:
        print("검색 결과가 없습니다.")
    return results

def get_pdf_list():
    """저장된 모든 PDF 파일 목록 조회"""
    results = collection.get()
    if results and results['metadatas']:
        pdf_files = sorted(set(meta['filename'] for meta in results['metadatas']))
        print("\n📚 저장된 PDF 파일 목록:")
        for i, pdf in enumerate(pdf_files, 1):
            print(f"{i}. {pdf}")
        return pdf_files
    return []

def get_pdf_sections(pdf_name):
    """특정 PDF의 섹션 목록 조회"""
    results = collection.get(
        where={"filename": pdf_name}
    )
    
    if results and results['metadatas']:
        sections = sorted(set(meta.get('title', '제목 없음') for meta in results['metadatas']))
        print(f"\n📑 {pdf_name} 섹션 목록:")
        for i, section in enumerate(sections, 1):
            print(f"{i}. {section}")
        return sections
    return []

def get_pdf_content(pdf_name, section_title=None):
    """PDF 내용 조회 (전체 또는 특정 섹션)"""
    where_clause = {"filename": pdf_name}
    if section_title:
        where_clause["title"] = section_title
    
    results = collection.get(where=where_clause)
    
    if not results or not results['metadatas']:
        print(f"❌ 내용을 찾을 수 없습니다.")
        return None
    
    # 청크 인덱스 순서대로 정렬
    chunks = list(zip(results['metadatas'], results['documents']))
    chunks.sort(key=lambda x: x[0].get('chunk_index', 0))
    
    print(f"\n📄 {pdf_name}")
    if section_title:
        print(f"섹션: {section_title}")
    print("=" * 80)
    
    for metadata, content in chunks:
        title = metadata.get('title', '제목 없음')
        if not section_title:  # 전체 내용 조회시 섹션 제목 표시
            print(f"\n## {title}")
        content_without_title = re.sub(r'<h\d>.*?</h\d>\n?', '', content, 1)
        print(content_without_title.strip())
        print("-" * 80)
    
    return chunks

def save_results_to_json(results, query_text, filename="search_results.json"):
    """검색 결과를 JSON 형식으로 저장"""
    output = {
        "query": query_text,
        "results": []
    }

    if results and results['ids'] and results['documents'] and results['metadatas']:
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            output["results"].append({
                "rank": i + 1,
                "id": results['ids'][0][i],
                "filename": metadata['filename'],
                "section": metadata.get('title', '제목 없음'),
                "document": results['documents'][0][i]
            })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"./results/{filename.replace('.json', '')}_{timestamp}.json"

        os.makedirs("results", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ 검색 결과가 저장되었습니다: {filepath}")
    else:
        print("⚠️ 저장할 검색 결과가 없습니다.")

def save_pdf_content(pdf_name, output_format="txt", section_title=None):
    """PDF 내용을 파일로 저장"""
    content = get_pdf_content(pdf_name, section_title)
    if not content:
        return
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    section_suffix = f"_{section_title}" if section_title else ""
    filename = f"{pdf_name}_{timestamp}{section_suffix}.{output_format}"
    filepath = os.path.join("results", filename)
    
    os.makedirs("results", exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        if output_format == "txt":
            for metadata, doc in content:
                title = metadata.get('title', '제목 없음')
                f.write(f"\n## {title}\n")
                content_without_title = re.sub(r'<h\d>.*?</h\d>\n?', '', doc, 1)
                f.write(content_without_title.strip() + "\n\n")
                f.write("-" * 80 + "\n")
        elif output_format == "json":
            json.dump({
                "filename": pdf_name,
                "section": section_title,
                "content": [{
                    "title": meta.get('title', '제목 없음'),
                    "text": re.sub(r'<h\d>.*?</h\d>\n?', '', doc, 1).strip()
                } for meta, doc in content]
            }, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 내용이 저장되었습니다: {filepath}")

def check_eligibility(user_info):
    """사용자 지원 자격 검증"""
    # 검색할 키워드 구성
    search_queries = [
        "지원자격",
        "신청자격",
        "자격요건",
        "소득기준",
        "자산기준",
        "연령제한",
        "나이제한",
        "자동차보유",
        "차량보유",
        "재학생",
        "졸업생",
        "취업여부",
        "수급자",
        "장애인"
    ]
    
    # 모든 PDF 파일 목록 가져오기
    pdf_files = get_pdf_list()
    eligibility_results = []
    
    for pdf_name in pdf_files:
        pdf_result = {
            "pdf_name": pdf_name,
            "eligible": True,
            "requirements": [],
            "matched_conditions": [],
            "unmet_conditions": []
        }
        
        # 각 검색 키워드에 대해 해당 PDF의 자격 조건 검색
        for query in search_queries:
            results = search_chunks(query, top_k=3, pdf_name=pdf_name)
            if results and results['documents']:
                for doc in results['documents'][0]:
                    requirement = analyze_requirement(doc, user_info)
                    if requirement:
                        pdf_result["requirements"].extend(requirement)
        
        # 자격 요건 분석 결과 정리
        pdf_result = analyze_eligibility(pdf_result, user_info)
        eligibility_results.append(pdf_result)
    
    return eligibility_results

def analyze_requirement(text, user_info):
    """텍스트에서 자격 요건 추출 및 분석"""
    requirements = []
    
    # 나이/연령 제한 확인
    birth_year = int(user_info["birth_date"][:4])
    current_year = datetime.datetime.now().year
    age = current_year - birth_year + 1  # 한국 나이
    
    age_patterns = [
        r'(\d+)세\s*(?:미만|이하|이상|초과)',
        r'만\s*(\d+)세\s*(?:미만|이하|이상|초과)',
        r'(\d+)년생\s*(?:이후|이전)'
    ]
    
    for pattern in age_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            requirements.append({
                "type": "age",
                "text": match.group(0),
                "value": int(match.group(1))
            })
    
    # 소득 기준 확인
    income_patterns = [
        r'소득\s*(\d+)%\s*이하',
        r'평균\s*소득\s*(\d+)%\s*이하',
        r'소득기준\s*(\d+)%\s*이하'
    ]
    
    for pattern in income_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            requirements.append({
                "type": "income",
                "text": match.group(0),
                "value": int(match.group(1))
            })
    
    # 자산 기준 확인
    asset_patterns = [
        r'총자산\s*(\d+)(?:만원|억원)',
        r'자산\s*(\d+)(?:만원|억원)'
    ]
    
    for pattern in asset_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            requirements.append({
                "type": "assets",
                "text": match.group(0),
                "value": convert_to_won(match.group(1), match.group(0))
            })
    
    # 차량 기준 확인
    if "차량" in text or "자동차" in text:
        requirements.append({
            "type": "car",
            "text": text,
            "value": None
        })
    
    # 수급자 관련 확인
    if "수급자" in text:
        requirements.append({
            "type": "welfare",
            "text": text,
            "value": None
        })
    
    return requirements

def analyze_eligibility(pdf_result, user_info):
    """자격 요건과 사용자 정보 비교 분석"""
    for req in pdf_result["requirements"]:
        if req["type"] == "age":
            birth_year = int(user_info["birth_date"][:4])
            current_year = datetime.datetime.now().year
            age = current_year - birth_year + 1
            
            if "이하" in req["text"] and age > req["value"]:
                pdf_result["eligible"] = False
                pdf_result["unmet_conditions"].append(f"나이 제한: {req['text']}")
            elif "이상" in req["text"] and age < req["value"]:
                pdf_result["eligible"] = False
                pdf_result["unmet_conditions"].append(f"나이 제한: {req['text']}")
            else:
                pdf_result["matched_conditions"].append(f"나이 조건 충족: {age}세")
        
        elif req["type"] == "income":
            user_income = int(user_info["income_range"].replace("% 이하", ""))
            if user_income > req["value"]:
                pdf_result["eligible"] = False
                pdf_result["unmet_conditions"].append(f"소득 기준: {req['text']}")
            else:
                pdf_result["matched_conditions"].append(f"소득 기준 충족: {user_income}%")
        
        elif req["type"] == "assets":
            if int(user_info["total_assets"]) > req["value"]:
                pdf_result["eligible"] = False
                pdf_result["unmet_conditions"].append(f"자산 기준: {req['text']}")
            else:
                pdf_result["matched_conditions"].append("자산 기준 충족")
        
        elif req["type"] == "car":
            if int(user_info["car_value"]) > 0:
                pdf_result["eligible"] = False
                pdf_result["unmet_conditions"].append("차량 보유 제한")
            else:
                pdf_result["matched_conditions"].append("차량 기준 충족")
    
    return pdf_result

def convert_to_won(value, unit):
    """금액 단위 변환"""
    value = int(value)
    if "억원" in unit:
        return value * 100000000
    elif "만원" in unit:
        return value * 10000
    return value

def get_eligible_programs(user_info):
    """사용자가 지원 가능한 프로그램 목록 조회"""
    results = check_eligibility(user_info)
    
    eligible_programs = []
    ineligible_programs = []
    
    for result in results:
        program_info = {
            "name": result["pdf_name"],
            "matched_conditions": result["matched_conditions"],
            "unmet_conditions": result["unmet_conditions"]
        }
        
        if result["eligible"]:
            eligible_programs.append(program_info)
        else:
            ineligible_programs.append(program_info)
    
    return {
        "eligible": eligible_programs,
        "ineligible": ineligible_programs
    }

def check_eligibility_for_specific_notice(pdf_name, user_info):
    """특정 공고문에 대한 사용자 지원 자격 검증"""
    
    # 자격 관련 섹션 검색
    eligibility_keywords = [
        "지원자격",
        "신청자격",
        "입주자격",
        "자격요건",
        "신청대상"
    ]
    
    eligibility_sections = []
    for keyword in eligibility_keywords:
        results = search_chunks(keyword, top_k=5, pdf_name=pdf_name)
        if results and results['documents'] and results['documents'][0]:
            eligibility_sections.extend(results['documents'][0])
    
    # 분석 결과를 저장할 딕셔너리
    analysis_result = {
        "pdf_name": pdf_name,
        "eligible": True,
        "matched_conditions": [],
        "unmet_conditions": [],
        "eligibility_details": {}
    }
    
    # 연령 조건 확인
    birth_year = int(user_info["birth_date"][:4])
    current_year = datetime.datetime.now().year
    age = current_year - birth_year + 1  # 한국 나이
    
    # 각 자격 조건 확인
    for section in eligibility_sections:
        # 1. 연령 제한 확인
        age_patterns = [
            r'(?:만\s*)?(\d+)세\s*(?:미만|이하|이상|초과)',
            r'(?:만\s*)?(\d+)세부터\s*(?:만\s*)?(\d+)세까지',
            r'(\d+)년(?:생|도)\s*(?:이후|이전)'
        ]
        
        for pattern in age_patterns:
            matches = re.finditer(pattern, section)
            for match in matches:
                age_text = match.group(0)
                if "이하" in age_text or "미만" in age_text:
                    age_limit = int(match.group(1))
                    if age > age_limit:
                        analysis_result["eligible"] = False
                        analysis_result["unmet_conditions"].append(f"연령 조건: {age_text} (현재 {age}세)")
                    else:
                        analysis_result["matched_conditions"].append(f"연령 조건 충족: {age}세")
                elif "이상" in age_text or "초과" in age_text:
                    age_limit = int(match.group(1))
                    if age < age_limit:
                        analysis_result["eligible"] = False
                        analysis_result["unmet_conditions"].append(f"연령 조건: {age_text} (현재 {age}세)")
                    else:
                        analysis_result["matched_conditions"].append(f"연령 조건 충족: {age}세")
        
        # 2. 소득 기준 확인
        income_patterns = [
            r'(?:소득|소득기준|평균소득)\s*(\d+)%\s*(?:이하|미만)',
            r'(?:소득|소득기준|평균소득)\s*(\d+)%\s*초과'
        ]
        
        user_income = int(user_info["income_range"].replace("% 이하", ""))
        for pattern in income_patterns:
            matches = re.finditer(pattern, section)
            for match in matches:
                income_text = match.group(0)
                income_limit = int(match.group(1))
                if "이하" in income_text or "미만" in income_text:
                    if user_income > income_limit:
                        analysis_result["eligible"] = False
                        analysis_result["unmet_conditions"].append(f"소득 기준: {income_text}")
                    else:
                        analysis_result["matched_conditions"].append(f"소득 기준 충족: {user_income}%")
        
        # 3. 자산 기준 확인
        asset_patterns = [
            r'(?:총자산|자산)\s*(\d+)(?:만원|억원)\s*(?:이하|미만)',
            r'(?:총자산|자산)\s*(\d+)(?:만원|억원)\s*초과'
        ]
        
        for pattern in asset_patterns:
            matches = re.finditer(pattern, section)
            for match in matches:
                asset_text = match.group(0)
                asset_value = convert_to_won(match.group(1), asset_text)
                if int(user_info["total_assets"]) > asset_value:
                    analysis_result["eligible"] = False
                    analysis_result["unmet_conditions"].append(f"자산 기준: {asset_text}")
                else:
                    analysis_result["matched_conditions"].append("자산 기준 충족")
        
        # 4. 차량 보유 기준 확인
        if "차량" in section or "자동차" in section:
            car_value = int(user_info["car_value"])
            if car_value > 0:
                analysis_result["eligible"] = False
                analysis_result["unmet_conditions"].append("차량 보유 제한")
            else:
                analysis_result["matched_conditions"].append("차량 기준 충족")
        
        # 5. 수급자 관련 확인
        if "수급자" in section and user_info["household_type"] == "생계·의료·주거급여 수급자 가구":
            analysis_result["matched_conditions"].append("수급자 가구 조건 충족")
        
        # 6. 대학생 관련 확인
        if "대학생" in section or "재학" in section:
            if user_info["university_status"] == "재학 중":
                analysis_result["matched_conditions"].append("대학생 조건 충족")
            else:
                analysis_result["unmet_conditions"].append("대학생(재학생) 조건 미충족")
    
    # 검증 결과 요약
    analysis_result["eligibility_details"] = {
        "total_conditions": len(analysis_result["matched_conditions"]) + len(analysis_result["unmet_conditions"]),
        "matched_count": len(analysis_result["matched_conditions"]),
        "unmet_count": len(analysis_result["unmet_conditions"])
    }
    
    return analysis_result

def print_eligibility_result(result):
    """자격 검증 결과 출력"""
    print("\n" + "=" * 80)
    print(f"📋 공고문: {result['pdf_name']}")
    print("=" * 80)
    
    if result["eligible"]:
        print("\n✅ 지원 가능합니다!")
    else:
        print("\n❌ 지원 자격이 충족되지 않습니다.")
    
    print("\n[충족된 조건]")
    for condition in result["matched_conditions"]:
        print(f"✓ {condition}")
    
    if result["unmet_conditions"]:
        print("\n[미충족된 조건]")
        for condition in result["unmet_conditions"]:
            print(f"✗ {condition}")
    
    print("\n[검증 결과 요약]")
    print(f"- 전체 조건 수: {result['eligibility_details']['total_conditions']}")
    print(f"- 충족된 조건 수: {result['eligibility_details']['matched_count']}")
    print(f"- 미충족된 조건 수: {result['eligibility_details']['unmet_count']}")
    print("=" * 80)

# 사용 예시
if __name__ == "__main__":
    # 사용자 정보
    user_info = {
        "birth_date": "990101",
        "gender": "남성",
        "university_status": "재학 중",
        "recent_graduate": "예",
        "employed": "아니오",
        "job_seeking": "아니오",
        "household_type": "생계·의료·주거급여 수급자 가구",
        "parents_own_house": "아니요",
        "disability_in_family": "예",
        "application_count": 2,
        "total_assets": 1000000,
        "car_value": 500000,
        "income_range": "100% 이하"
    }
    
    # 특정 공고문 선택
    pdf_name = "[마을과집]SH특화형 매입임대주택(청년) 입주자 모집 공고문_20250307.pdf"
    
    # 자격 검증
    result = check_eligibility_for_specific_notice(pdf_name, user_info)
    
    # 결과 출력
    print_eligibility_result(result)