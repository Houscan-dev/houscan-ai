from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from typing import Dict, Any
from pydantic import BaseModel

# RAG 챗봇 임포트
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_chatbot import RAGChatbot, get_available_pdfs

app = FastAPI(title="Houscan AI Server")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    pdf_name: str
    user_info: Dict[str, Any] = None

# 전역 챗봇 인스턴스 저장소
chatbots = {}

@app.get("/api/parsed-info")
async def get_parsed_info() -> Dict[str, Any]:
    """추출된 주택 정보를 반환하는 엔드포인트"""
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 주택 정보 JSON 파일 읽기
        with open(os.path.join(base_dir, "extracted_housing_info/housing_info.json"), "r", encoding="utf-8") as f:
            housing_info = json.load(f)
        
        # 주의사항 JSON 파일 읽기
        with open(os.path.join(base_dir, "extracted_precautions/precautions.json"), "r", encoding="utf-8") as f:
            precautions = json.load(f)
        
        # 우선순위 및 점수 JSON 파일 읽기
        with open(os.path.join(base_dir, "extracted_priority_and_score/priority_and_score.json"), "r", encoding="utf-8") as f:
            priority_score = json.load(f)
        
        # 거주기간 JSON 파일 읽기
        with open(os.path.join(base_dir, "extracted_residence_period/residence_period.json"), "r", encoding="utf-8") as f:
            residence_period = json.load(f)
        
        return {
            "housing_info": housing_info,
            "precautions": precautions,
            "priority_score": priority_score,
            "residence_period": residence_period
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")

@app.get("/api/available-pdfs")
async def get_pdfs() -> Dict[str, list]:
    """사용 가능한 PDF 파일 목록을 반환하는 엔드포인트"""
    try:
        pdfs = get_available_pdfs()
        return {"pdfs": pdfs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 목록 조회 중 오류 발생: {str(e)}")

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> Dict[str, Any]:
    """챗봇 응답을 생성하는 엔드포인트"""
    try:
        # PDF 이름으로 챗봇 인스턴스 가져오기 또는 생성
        if request.pdf_name not in chatbots:
            chatbots[request.pdf_name] = RAGChatbot(request.pdf_name, request.user_info)
        else:
            # 사용자 정보가 제공된 경우 업데이트
            if request.user_info:
                chatbots[request.pdf_name].update_user_info(request.user_info)
        
        # 챗봇 응답 생성
        response = chatbots[request.pdf_name].chat(request.question)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 응답 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 