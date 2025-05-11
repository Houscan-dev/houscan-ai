"""
애플리케이션 설정
"""
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 기본 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# 디렉토리 생성
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# API 설정
API_V1_PREFIX = "/api/v1"
PROJECT_NAME = "Houscan AI API"
VERSION = "1.0.0"

# 모델 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo") 