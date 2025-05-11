import os
import torch
import json
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 모델 로드
try:
    logger.info("모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    model = AutoModel.from_pretrained("BAAI/bge-m3")
    model.eval()
    logger.info("모델 로딩 완료")
except Exception as e:
    logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
    raise

# 처리된 문서 폴더 경로
processed_folder = "./processed_docs"
if not os.path.exists(processed_folder):
    logger.error(f"처리된 문서 폴더가 존재하지 않습니다: {processed_folder}")
    raise FileNotFoundError(f"처리된 문서 폴더가 존재하지 않습니다: {processed_folder}")

# ChromaDB 클라이언트 설정
try:
    logger.info("ChromaDB 클라이언트 초기화 중...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="processed_chunks",
        metadata={"description": "처리된 문서 청크 임베딩 저장소"}
    )
    logger.info("ChromaDB 클라이언트 초기화 완료")
except Exception as e:
    logger.error(f"ChromaDB 초기화 중 오류 발생: {str(e)}")
    raise

def get_embedding(text, max_length=512):
    """CLS 토큰 벡터 추출 함수"""
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length, 
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
        return embedding.squeeze().tolist()
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        return None

def process_batch(batch_texts, batch_ids, batch_metadatas):
    """배치 단위로 임베딩을 생성하고 ChromaDB에 저장하는 함수"""
    try:
        embeddings = []
        for text in batch_texts:
            embedding = get_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
        
        if embeddings:
            collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            logger.info(f"배치 처리 완료: {len(batch_texts)}개 문서")
            return True
        return False
    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {str(e)}")
        return False

def process_json_file(json_path, batch_size=10):
    """JSON 파일의 청크들을 처리하는 함수"""
    try:
        logger.info(f"JSON 파일 처리 중: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        filename = data["filename"]
        chunks = data["chunks"]
        
        # 배치 처리
        batch_texts = []
        batch_ids = []
        batch_metadatas = []
        
        for chunk in chunks:
            metadata = {
                "filename": filename,
                "chunk_index": chunk["chunk_index"],
                "total_chunks": data["total_chunks"],
                "processed_date": data["processed_date"]
            }
            
            batch_texts.append(chunk["content"])
            batch_ids.append(f"{filename}_{chunk['chunk_index']}")
            batch_metadatas.append(metadata)
            
            if len(batch_texts) >= batch_size:
                process_batch(batch_texts, batch_ids, batch_metadatas)
                batch_texts = []
                batch_ids = []
                batch_metadatas = []
        
        # 남은 배치 처리
        if batch_texts:
            process_batch(batch_texts, batch_ids, batch_metadatas)
        
        logger.info(f"JSON 파일 처리 완료: {json_path}")
        
    except Exception as e:
        logger.error(f"JSON 파일 처리 중 오류 발생 ({json_path}): {str(e)}")

def process_all_json_files():
    """모든 JSON 파일을 처리하는 메인 함수"""
    try:
        # JSON 파일 목록 가져오기
        json_files = [f for f in os.listdir(processed_folder) if f.endswith('_processed.json')]
        if not json_files:
            logger.warning("처리할 JSON 파일이 없습니다.")
            return
        
        logger.info(f"총 {len(json_files)}개의 JSON 파일을 처리합니다.")
        
        # 각 JSON 파일 처리
        for json_file in tqdm(json_files, desc="JSON 파일 처리 중"):
            json_path = os.path.join(processed_folder, json_file)
            process_json_file(json_path)
        
        logger.info("모든 JSON 파일 처리 완료")
        
    except Exception as e:
        logger.error(f"JSON 파일 처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    process_all_json_files()
