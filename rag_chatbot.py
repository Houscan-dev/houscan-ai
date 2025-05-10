import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
import logging
from transformers import AutoTokenizer, AutoModel
import torch
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, pdf_name: str):
        """
        RAG 기반 챗봇 초기화
        
        Args:
            pdf_name (str): PDF 파일 이름
        """
        self.pdf_name = pdf_name
        
        # 임베딩 모델 초기화 (기존 코드와 동일한 모델 사용)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            self.model = AutoModel.from_pretrained("BAAI/bge-m3")
            if torch.cuda.is_available():
                self.model.to("cuda")
            self.model.eval()
            
            # LangChain 임베딩 래퍼
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            logger.error(f"임베딩 모델 로딩 중 오류 발생: {str(e)}")
            raise
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # 벡터 스토어 초기화 (기존 컬렉션 사용)
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="processed_chunks",
            embedding_function=self.embeddings
        )
        
        # OpenAI API 키 확인
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # GPT 모델 초기화
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            openai_api_key=openai_api_key
        )
        
        # 대화 메모리 초기화
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # RAG 체인 초기화
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 3,
                    "filter": {"filename": pdf_name}  # 특정 PDF 파일만 검색
                }
            ),
            memory=self.memory,
            return_source_documents=True
        )
    
    def chat(self, query: str) -> Dict:
        """
        사용자 질문에 대한 응답 생성
        
        Args:
            query (str): 사용자 질문
            
        Returns:
            Dict: 응답과 관련 문서 정보를 포함한 딕셔너리
        """
        try:
            # RAG 체인을 통한 응답 생성 (invoke 메서드 사용)
            result = self.qa_chain.invoke({"question": query})
            
            # 응답 포맷팅
            response = {
                "answer": result["answer"],
                "sources": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result["source_documents"]
                ]
            }
            
            return response
            
        except Exception as e:
            logger.error(f"채팅 처리 중 오류 발생: {str(e)}")
            return {
                "answer": "죄송합니다. 응답을 생성하는 중에 오류가 발생했습니다.",
                "sources": []
            }

def get_available_pdfs() -> List[str]:
    """
    사용 가능한 PDF 파일 목록 반환
    
    Returns:
        List[str]: PDF 파일 이름 목록
    """
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("processed_chunks")
        results = collection.get()
        unique_filenames = set(meta['filename'] for meta in results['metadatas'])
        return list(unique_filenames)
    except Exception as e:
        logger.error(f"PDF 목록 조회 중 오류 발생: {str(e)}")
        return []

if __name__ == "__main__":
    # 사용 가능한 PDF 목록 가져오기
    pdfs = get_available_pdfs()
    print("\n=== 사용 가능한 공고문 목록 ===")
    for i, pdf in enumerate(pdfs, 1):
        print(f"{i}. {pdf}")
    
    if pdfs:
        # 첫 번째 공고문 선택
        selected_pdf = pdfs[0]
        print(f"\n선택된 공고문: {selected_pdf}")
        
        # 챗봇 초기화
        print("\n챗봇 초기화 중...")
        chatbot = RAGChatbot(selected_pdf)
        print("챗봇 초기화 완료!")
        
        # 질문하기
        question = "이 공고문의 주요 내용이 뭐야?"
        print(f"\n질문: {question}")
        
        response = chatbot.chat(question)
        
        # 응답 출력
        print("\n=== 응답 ===")
        print(response["answer"])
        
        # 참고 문서 출력
        print("\n=== 참고 문서 ===")
        for i, source in enumerate(response["sources"], 1):
            print(f"\n[참고 문서 {i}]")
            print("내용:", source["content"])
            print("메타데이터:", source["metadata"])
    else:
        print("\n사용 가능한 공고문이 없습니다.") 