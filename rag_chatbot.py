import os
from typing import List, Dict, Optional
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
import json

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(self, pdf_name: str, user_info: Optional[Dict] = None):
        """
        RAG 기반 챗봇 초기화
        
        Args:
            pdf_name (str): PDF 파일 이름
            user_info (Dict, optional): 사용자 정보를 담은 딕셔너리
        """
        self.pdf_name = pdf_name
        self.user_info = user_info or {}
        
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
                    "k": 5,  # 검색할 문서 수 증가
                    "filter": {"filename": pdf_name}  # 특정 PDF 파일만 검색
                }
            ),
            memory=self.memory,
            return_source_documents=True,
            verbose=True  # 디버깅을 위한 상세 로그 활성화
        )
    
    def update_user_info(self, user_info: Dict):
        """
        사용자 정보 업데이트
        
        Args:
            user_info (Dict): 새로운 사용자 정보
        """
        self.user_info.update(user_info)
    
    def chat(self, query: str) -> Dict:
        """
        사용자 질문에 대한 응답 생성
        
        Args:
            query (str): 사용자 질문
            
        Returns:
            Dict: 응답과 관련 문서 정보를 포함한 딕셔너리
        """
        try:
            # 사용자 정보가 필요한 질문인지 확인
            user_info_keywords = [
                "내가", "저는", "제가", "나의", "저의", "본인", "나", "저",
                "신청", "지원", "해당", "자격", "조건", "요건", "적합", "부적합"
            ]
            
            # 사용자 정보가 필요한 질문인 경우에만 컨텍스트 추가
            needs_user_info = any(keyword in query for keyword in user_info_keywords)
            
            # 시스템 프롬프트 추가
            system_prompt = """당신은 주택 공고문 전문가입니다. 다음 지침을 따라 답변해주세요:
            1. 공고문의 내용을 정확하게 이해하고 답변하세요.
            2. 지원 자격 요건은 반드시 공고문에 명시된 내용만 언급하세요.
            3. 불확실한 내용은 추측하지 말고, 공고문에 명시된 내용만 답변하세요.
            4. 답변은 간결하고 명확하게 작성하세요.
            5. 필요한 경우 공고문의 구체적인 내용을 인용하세요.
            """
            
            context = system_prompt + "\n\n"
            if needs_user_info and self.user_info:
                context += f"""
                사용자 정보:
                - 생년월일: {self.user_info.get('birth_date', '정보 없음')}
                - 성별: {self.user_info.get('gender', '정보 없음')}
                - 대학생 여부: {self.user_info.get('university_status', '정보 없음')}
                - 최근 졸업생 여부: {self.user_info.get('recent_graduate', '정보 없음')}
                - 취업 여부: {self.user_info.get('employed', '정보 없음')}
                - 구직 여부: {self.user_info.get('job_seeking', '정보 없음')}
                - 가구 유형: {self.user_info.get('household_type', '정보 없음')}
                - 부모 주택 보유 여부: {self.user_info.get('parents_own_house', '정보 없음')}
                - 가족 중 장애인 여부: {self.user_info.get('disability_in_family', '정보 없음')}
                - 신청 횟수: {self.user_info.get('application_count', '정보 없음')}회
                - 총 자산: {self.user_info.get('total_assets', '정보 없음')}원
                - 차량 가액: {self.user_info.get('car_value', '정보 없음')}원
                - 소득 범위: {self.user_info.get('income_range', '정보 없음')}
                - 혼인 상태: {self.user_info.get('marital_status', '정보 없음')}
                - 노숙인 여부: {self.user_info.get('is_homeless', '정보 없음')}
                
                위 사용자 정보를 바탕으로 다음 질문에 답변해주세요:
                """
            
            # RAG 체인을 통한 응답 생성
            result = self.qa_chain.invoke({
                "question": context + query if context else query
            })
            
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
        
        # 테스트용 사용자 정보
        user_info = {
            "birth_date": "19990101",
            "gender": "남성",
            "university_status": "재학 중",
            "recent_graduate": "예",
            "employed": "아니오",
            "job_seeking": "아니오",
            "household_type": "생계·의료·주거급여 수급자 가구",
            "parents_own_house": "아니요",
            "disability_in_family": "예",
            "application_count": 2,
            "total_assets": 100000000,
            "car_value": 5000000,
            "income_range": "100% 이하"
        }
        
        # 챗봇 초기화 (사용자 정보 포함)
        print("\n챗봇 초기화 중...")
        chatbot = RAGChatbot(selected_pdf, user_info)
        print("챗봇 초기화 완료!")
        
        # 질문하기
        question = "이 공고의 지원자격요건이 뭐지?"
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