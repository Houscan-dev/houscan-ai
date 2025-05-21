from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_chatbot import RAGChatbot, get_available_pdfs
import logging
import json
import os
from datetime import datetime, timedelta
import threading

app = Flask(__name__)
# CORS 설정 - 모든 도메인 허용, credentials 없이
CORS(app, 
     origins="*", 
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotManager:
    def __init__(self):
        self.chatbots = {}  # PDF별 챗봇 인스턴스
        self.last_used = {}  # 마지막 사용 시간
        self.lock = threading.Lock()  # 스레드 안전성을 위한 락
        self.max_idle_time = timedelta(hours=1)  # 최대 유휴 시간
        
    def get_chatbot(self, pdf_name: str) -> RAGChatbot:
        """PDF 이름에 해당하는 챗봇 인스턴스를 반환"""
        with self.lock:
            # 챗봇이 없거나 만료된 경우 새로 생성
            if pdf_name not in self.chatbots or self._is_expired(pdf_name):
                self.chatbots[pdf_name] = RAGChatbot(pdf_name)
                logger.info(f"새로운 챗봇 인스턴스 생성: {pdf_name}")
            
            # 마지막 사용 시간 업데이트
            self.last_used[pdf_name] = datetime.now()
            
            return self.chatbots[pdf_name]
    
    def _is_expired(self, pdf_name: str) -> bool:
        """챗봇 인스턴스가 만료되었는지 확인"""
        if pdf_name not in self.last_used:
            return True
        return datetime.now() - self.last_used[pdf_name] > self.max_idle_time
    
    def cleanup_expired(self):
        """만료된 챗봇 인스턴스 정리"""
        with self.lock:
            expired = [pdf_name for pdf_name in self.chatbots if self._is_expired(pdf_name)]
            for pdf_name in expired:
                del self.chatbots[pdf_name]
                del self.last_used[pdf_name]
                logger.info(f"만료된 챗봇 인스턴스 제거: {pdf_name}")

# 챗봇 매니저 인스턴스 생성
chatbot_manager = ChatbotManager()

# 주기적으로 만료된 챗봇 정리
def cleanup_task():
    while True:
        chatbot_manager.cleanup_expired()
        threading.Event().wait(300)  # 5분마다 실행

# 백그라운드에서 정리 작업 시작
cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
cleanup_thread.start()

@app.route('/api/available-pdfs', methods=['GET'])
def get_pdfs():
    """사용 가능한 PDF 파일 목록을 반환하는 엔드포인트"""
    try:
        pdfs = get_available_pdfs()
        return jsonify({
            'status': 'success',
            'pdfs': pdfs
        })
    except Exception as e:
        logger.error(f"PDF 목록 조회 중 오류 발생: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"PDF 목록 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    PDF 문서에 대한 질문을 처리하고 응답을 반환하는 엔드포인트
    
    요청 형식:
    {
        "pdf_name": "문서명.pdf",  # 필수: 질문할 PDF 문서 이름
        "query": "질문 내용"       # 필수: 사용자의 질문
    }
    
    응답 형식:
    {
        "status": "success",
        "data": "챗봇의 응답"
    }
    또는
    {
        "status": "error",
        "message": "에러 메시지"
    }
    """
    try:
        # JSON 요청 데이터 파싱
        data = request.get_json()
        pdf_name = data.get('pdf_name')
        query = data.get('query')
        
        # 필수 파라미터 검증
        if not pdf_name:
            return jsonify({
                'status': 'error',
                'message': 'PDF 이름이 제공되지 않았습니다.'
            }), 400
            
        if not query:
            return jsonify({
                'status': 'error',
                'message': '질문이 제공되지 않았습니다.'
            }), 400
            
        # ChatbotManager를 통해 해당 PDF의 챗봇 인스턴스 가져오기
        # - 없으면 새로 생성
        # - 있으면 기존 인스턴스 사용
        chatbot = chatbot_manager.get_chatbot(pdf_name)
        
        # 챗봇을 통해 질문에 대한 응답 생성
        response = chatbot.chat(query)
        
        # 성공 응답 반환
        return jsonify({
            'status': 'success',
            'data': response
        })
    except Exception as e:
        # 에러 발생 시 로깅 및 에러 응답 반환
        logger.error(f"채팅 처리 중 오류 발생: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)