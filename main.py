"""
Houscan AI 메인 애플리케이션
"""
from api.app import app
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Houscan AI 서버를 시작합니다...")
    app.run(host="0.0.0.0", port=5000, debug=True) 