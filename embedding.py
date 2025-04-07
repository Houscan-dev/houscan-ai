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

# chunk 폴더 경로
chunk_folder = "./chunks"

# 새 클라이언트 생성 방식
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 컬렉션 가져오기/생성
collection = chroma_client.get_or_create_collection(name="my_chunks")

# 함수: CLS 토큰 벡터 추출
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.squeeze().tolist()

# 모든 txt 파일 처리 후 Chroma에 추가
for idx, filename in enumerate(sorted(os.listdir(chunk_folder))):
    if filename.endswith(".txt"):
        with open(os.path.join(chunk_folder, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
            embedding = get_embedding(text)

            # Chroma에 추가
            collection.add(
                ids=[f"doc_{idx}"],  # 유니크한 ID 필수
                embeddings=[embedding],
                documents=[text],
                metadatas=[{"filename": filename}]
            )
