## ※ 일부 수정 중입니다 ※


# 하우스캔 AI 파트 리포지토리입니다.


## 프로젝트 구조

### 1. 데이터 전처리
- `data/pdfs/`: 청약공고문 원본 PDF 파일들
- `scripts/document_parsing.py`: 문서 파싱, 타이틀 태깅, 청크로 구조화
- `data/processed/`: 청크 구조화까지 완료된 JSON 파일들 저장 폴더
- `scripts/embedding.py`: `data/processed` 내의 파일들을 임베딩해서 벡터DB에 저장
- `chroma_db/`: 벡터DB, 임베딩 완료된 내용이 저장됨

### 2. 정보 추출
`data/extracted/` 디렉토리에 다음 정보들이 추출되어 저장됩니다:
- `criteria/`: 신청자격
- `housing_info/`: 공급주택정보
- `precautions/`: 유의사항
- `priority_score/`: 우선순위 및 가점사항
- `residence_period/`: 거주기간
- `schedule/`: 모집일정

### 3. 데이터 처리 스크립트
`scripts/` 디렉토리에 다음 스크립트들이 포함되어 있습니다:
- `extract_criteria.py`: 신청자격 추출
- `extract_housing_info.py`: 공급주택정보 추출
- `extract_precautions.py`: 유의사항 추출
- `extract_priority_and_score.py`: 우선순위 및 가점사항 추출
- `extract_residence_period.py`: 거주기간 추출
- `extract_schedule.py`: 모집일정 추출
- `fix_spacing.py`: 띄어쓰기 교정
- `format_criteria.py`: criteria 텍스트를 JSON 형식으로 변환

### 4. 챗봇 시스템
- `rag_chatbot.py`: 챗봇 핵심 기능 정의
- `app.py`: 프론트엔드와 상호작용 (Flask 사용)
- `test_chatbot.py`: 챗봇 테스트용 코드

### 5. 판단 로직
- `housing_eligibility_analyzer.py`: 우선순위 및 신청자격 판단 로직

### 6. 기타
- `old_extracted/`: 이전 버전의 정보 추출 결과 (현재 사용하지 않음)
