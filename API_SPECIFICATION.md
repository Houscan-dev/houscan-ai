# Houscan AI API 명세서

## 기본 정보
- **기본 URL**: `http://localhost:5000`
- **응답 형식**: JSON
- **인증**: 없음

| 기능                | HTTP Method | Endpoint                | 설명                                      | Request Body                                                                                                    | Response (성공)                                                                                                                                                                                                 | Response (실패)                                                                                                 |
|---------------------|-------------|-------------------------|-------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 사용 가능 PDF 목록 조회 | GET         | /api/available-pdfs     | 현재 챗봇이 지원하는 공고문(PDF) 목록 조회    | N/A                                                                                                              | 200 OK<br>{<br> "status": "success",<br> "pdfs": ["공고문1.pdf", "공고문2.pdf", ...]<br>}                                                              | 500 Internal Server Error<br>{<br> "status": "error",<br> "message": "PDF 목록 조회 중 오류 발생: ..."<br>}      |
| 챗봇 질의응답       | POST        | /api/chat               | 챗봇에게 질문을 보내고 답변 받기             | {<br> "pdf_name": "공고문_파일명.pdf",<br> "query": "이 공고의 지원자격요건이 뭐지?"<br>}                         | 200 OK<br>{<br> "status": "success",<br> "data": {<br> "answer": "응답 내용",<br> "sources": [<br> {<br> "content": "참고 문서 내용",<br> "metadata": { "filename": "공고문_파일명.pdf", ... }<br> }<br> ]<br> }<br>} | 400/500 Bad Request<br>{<br> "status": "error",<br> "message": "PDF 이름이 제공되지 않았습니다." 또는 "질문이 제공되지 않았습니다." 또는 기타 에러 메시지<br>}                              |

---

### 상세 예시

#### 1. 사용 가능 PDF 목록 조회
```
GET /api/available-pdfs
```
성공:
```json
{
  "status": "success",
  "pdfs": ["공고문1.pdf", "공고문2.pdf"]
}
```
실패:
```json
{
  "status": "error",
  "message": "PDF 목록 조회 중 오류 발생: ..."
}
```

#### 2. 챗봇 질의응답
```
POST /api/chat
Content-Type: application/json
{
  "pdf_name": "공고문_파일명.pdf",
  "query": "이 공고의 지원자격요건이 뭐지?"
}
```
성공:
```json
{
  "status": "success",
  "data": {
    "answer": "응답 내용",
    "sources": [
      {
        "content": "참고 문서 내용",
        "metadata": {
          "filename": "공고문_파일명.pdf"
        }
      }
    ]
  }
}
```
실패:
```json
{
  "status": "error",
  "message": "PDF 이름이 제공되지 않았습니다."
}
```
또는
```json
{
  "status": "error",
  "message": "질문이 제공되지 않았습니다."
}
```
또는 기타 에러 메시지 