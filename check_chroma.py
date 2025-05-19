import chromadb
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(path="./chroma_db")

# 모든 컬렉션 목록 가져오기
collections = client.list_collections()
logger.info(f"컬렉션 목록: {[col.name for col in collections]}")

# processed_chunks 컬렉션 확인
try:
    collection = client.get_collection("processed_chunks")
    results = collection.get()
    
    # 메타데이터에서 고유한 파일명 추출
    unique_filenames = set(meta['filename'] for meta in results['metadatas'])
    logger.info(f"저장된 PDF 파일 목록: {list(unique_filenames)}")
    
    # 전체 문서 수 확인
    logger.info(f"전체 문서 수: {len(results['ids'])}")
    
except Exception as e:
    logger.error(f"컬렉션 조회 중 오류 발생: {str(e)}") 