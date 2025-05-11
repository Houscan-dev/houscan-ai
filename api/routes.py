from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any
from services.eligibility import EligibilityService
from services.schedule import ScheduleService
import os
import uuid

router = APIRouter()
eligibility_service = EligibilityService()
schedule_service = ScheduleService()

@router.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    """
    문서 분석 API 엔드포인트
    """
    try:
        # 임시 파일 저장
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = f"data/temp/{uuid.uuid4()}{file_ext}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # 문서 분석
        eligibility_result = await eligibility_service.analyze_eligibility(temp_path)
        schedule_result = await schedule_service.analyze_schedule(temp_path)
        
        # 임시 파일 삭제
        os.remove(temp_path)
        
        return {
            "status": "success",
            "eligibility": eligibility_result,
            "schedule": schedule_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-eligibility")
async def check_eligibility(criteria: Dict[str, Any]):
    """
    자격요건 확인 API 엔드포인트
    """
    try:
        # TODO: 자격요건 확인 로직 구현
        return {"status": "success", "message": "자격요건 확인이 완료되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 