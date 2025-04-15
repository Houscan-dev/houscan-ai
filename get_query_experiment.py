import os
import torch
import json
import csv
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import datetime

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")
model.eval()

# 새 클라이언트 생성 방식
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection(name="my_chunks")


# 함수: CLS 토큰 벡터 추출
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().tolist()

def search_chunks(query_text, top_k=3):
    # 쿼리 문장 임베딩 (BGE 스타일: Instruction 추가)
    query_embedding = get_embedding("Represent this sentence for retrieval: " + query_text)

    # ChromaDB에서 유사한 청크 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 결과 출력
    print(f"\n🔍 '{query_text}'에 대한 검색 결과 (Top {top_k}):\n")
    if results and results['ids'] and results['documents'] and results['metadatas']:
        for i in range(len(results['ids'][0])):
            print(f"[{i+1}] 문서 ID: {results['ids'][0][i]}")
            print(f"파일명: {results['metadatas'][0][i]['filename']}")
            print(f"내용: {results['documents'][0][i][:150]}...")  # 처음 150자만 출력
            print("-" * 60)
    else:
        print("검색 결과가 없습니다.")
    return results  # 검색 결과를 반환하도록 수정


def save_results_to_json(results, query_text, filename="search_results.json"):
    output = {
        "query": query_text,
        "results": []
    }

    if results and results['ids'] and results['documents'] and results['metadatas']:
        for i in range(len(results['ids'][0])):
            output["results"].append({
                "rank": i + 1,
                "id": results['ids'][0][i],
                "filename": results['metadatas'][0][i]['filename'],
                "document": results['documents'][0][i]
            })

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"./results/{filename.replace('.json', '')}_{timestamp}.json"

        os.makedirs("results", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ JSON 형식으로 검색 결과가 저장되었습니다: {filepath}")
    else:
        print("⚠️ 저장할 검색 결과가 없습니다.")

def save_results_to_csv(results, query_text, filename="search_results.csv"):
    if not results or not results['ids'] or not results['documents'] or not results['metadatas']:
        print("⚠️ 저장할 검색 결과가 없습니다.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"./results/{filename.replace('.csv', '')}_{timestamp}.csv"

    os.makedirs("results", exist_ok=True)
    with open(filepath, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["순위", "문서 ID", "파일명", "내용"])
        
        for i in range(len(results['ids'][0])):
            writer.writerow([
                i + 1,
                results['ids'][0][i],
                results['metadatas'][0][i]['filename'],
                results['documents'][0][i]
            ])

    print(f"✅ CSV 형식으로 검색 결과가 저장되었습니다: {filepath}")

def save_results_to_txt(results, query_text, filename="search_results.txt"):
    if not results or not results['ids'] or not results['documents'] or not results['metadatas']:
        print("⚠️ 저장할 검색 결과가 없습니다.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"./results/{filename.replace('.txt', '')}_{timestamp}.txt"

    os.makedirs("results", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"검색 쿼리: {query_text}\n\n")
        
        for i in range(len(results['ids'][0])):
            f.write(f"순위: {i+1}\n")
            f.write(f"문서 ID: {results['ids'][0][i]}\n")
            f.write(f"파일명: {results['metadatas'][0][i]['filename']}\n")
            f.write(f"내용: {results['documents'][0][i]}\n")
            f.write("-" * 60 + "\n")

    print(f"✅ TXT 형식으로 검색 결과가 저장되었습니다: {filepath}")

def save_results_to_markdown(results, query_text, filename="search_results.md"):
    if not results or not results['ids'] or not results['documents'] or not results['metadatas']:
        print("⚠️ 저장할 검색 결과가 없습니다.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"./results/{filename.replace('.md', '')}_{timestamp}.md"

    os.makedirs("results", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# 검색 결과\n\n")
        f.write(f"**검색 쿼리**: {query_text}\n\n")
        
        for i in range(len(results['ids'][0])):
            f.write(f"## {i+1}번째 결과\n")
            f.write(f"- **문서 ID**: {results['ids'][0][i]}\n")
            f.write(f"- **파일명**: {results['metadatas'][0][i]['filename']}\n")
            f.write(f"- **내용**:\n\n")
            f.write(f"```\n{results['documents'][0][i]}\n```\n\n")
            f.write("---\n\n")

    print(f"✅ Markdown 형식으로 검색 결과가 저장되었습니다: {filepath}")

def save_results(results, query_text, format="all"):
    """모든 형식으로 결과를 저장하는 함수"""
    if format == "all" or format == "json":
        save_results_to_json(results, query_text)
    if format == "all" or format == "csv":
        save_results_to_csv(results, query_text)
    if format == "all" or format == "txt":
        save_results_to_txt(results, query_text)
    if format == "all" or format == "md":
        save_results_to_markdown(results, query_text)

# 사용 예시
query = "청약 신청 자격 알려줘"
search_results = search_chunks(query, top_k=5)
save_results(search_results, query, format="all")  # 모든 형식으로 저장
# 또는 특정 형식만 저장
# save_results(search_results, query, format="json")  # JSON만 저장
# save_results(search_results, query, format="csv")   # CSV만 저장
# save_results(search_results, query, format="txt")   # TXT만 저장
# save_results(search_results, query, format="md")    # Markdown만 저장