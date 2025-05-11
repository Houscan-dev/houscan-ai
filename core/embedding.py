"""
문서 임베딩 관련 핵심 기능
"""
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

class DocumentEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client(Settings(
            persist_directory="data/chroma_db",
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("documents")

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트에 대한 임베딩 생성
        """
        return self.model.encode(texts).tolist()

    def store_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        문서를 벡터 DB에 저장
        """
        embedding = self.create_embeddings([text])[0]
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[document_id]
        )

    def search_similar(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        유사한 문서 검색
        """
        query_embedding = self.create_embeddings([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return [
            {
                "id": id,
                "text": doc,
                "metadata": meta,
                "distance": dist
            }
            for id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ] 