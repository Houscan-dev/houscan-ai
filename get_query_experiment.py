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

# 사용 예시
if __name__ == "__main__":
    # 1. PDF 목록 확인
    print("\n=== 저장된 PDF 파일 목록 ===")
    pdf_files = get_pdf_list()
    if not pdf_files:
        print("저장된 PDF 파일이 없습니다.")
        exit()

    # 2. 특정 PDF 선택 (예시)
    selected_pdf = pdf_files[0]
    
    # 3. 선택된 PDF의 섹션 목록 확인
    print(f"\n=== {selected_pdf} 섹션 목록 ===")
    sections = get_pdf_sections(selected_pdf)
    
    # 4. 검색 예시
    query = "청약 신청 자격 알려줘"
    
    # 4.1 전체 PDF에서 검색
    print("\n=== 전체 PDF 검색 ===")
    all_results = search_chunks(query, top_k=3)
    
    # 4.2 특정 PDF에서만 검색
    print(f"\n=== {selected_pdf} 검색 ===")
    pdf_results = search_chunks(query, top_k=3, pdf_name=selected_pdf)
    
    # 5. 특정 PDF의 전체 내용 저장
    save_pdf_content(selected_pdf, output_format="txt")
    
    # 6. 특정 PDF의 특정 섹션 내용 저장 (섹션이 있는 경우)
    if sections:
        save_pdf_content(selected_pdf, output_format="txt", section_title=sections[0])
    
    # 7. 검색 결과 저장
    save_results_to_json(all_results, query)