import os
import torch
import json
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel

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
    for i in range(len(results['ids'][0])):
        print(f"[{i+1}] 문서 ID: {results['ids'][0][i]}")
        print(f"파일명: {results['metadatas'][0][i]['filename']}")
        print(f"내용: {results['documents'][0][i][:150]}...")  # 처음 150자만 출력
        print("-" * 60)


search_chunks("강남구 무순위 청약 조건 알려줘", top_k=5)