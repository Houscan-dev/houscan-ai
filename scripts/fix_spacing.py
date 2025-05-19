# 띄어쓰기 이상한 파일들 다시 교정정

import os
import json
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def fix_spacing(text):
    """OpenAI API를 사용하여 텍스트의 띄어쓰기를 교정합니다."""
    try:
        # 숫자나 특수문자가 포함된 경우 건너뛰기
        if any(char.isdigit() for char in text) or any(not char.isalnum() and not char.isspace() for char in text):
            return text
            
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 한국어 띄어쓰기 교정 전문가입니다. 주어진 텍스트의 띄어쓰기만 교정하고 다른 내용은 전혀 수정하지 마세요. 문장의 의미, 구조, 특수문자, 숫자 등은 그대로 유지하고 오직 띄어쓰기만 수정하세요."},
                {"role": "user", "content": text}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing text: {e}")
        return text

def process_json_file(file_path):
    """JSON 파일을 읽고 priority_criteria의 criteria 필드의 띄어쓰기만 교정합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # priority_criteria가 리스트인 경우 처리
        if isinstance(data, dict) and "priority_criteria" in data:
            for item in data["priority_criteria"]:
                if isinstance(item, dict) and "criteria" in item:
                    if isinstance(item["criteria"], list):
                        # criteria가 리스트인 경우 각 항목의 띄어쓰기를 교정
                        item["criteria"] = [fix_spacing(c) for c in item["criteria"]]
                    elif isinstance(item["criteria"], str):
                        # criteria가 문자열인 경우 띄어쓰기를 교정
                        item["criteria"] = fix_spacing(item["criteria"])
        
        # 교정된 내용을 원본 파일에 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    base_dir = Path("data/extracted")
    
    # 처리할 폴더 목록
    target_folders = ["priority_score"]
    
    # 각 폴더의 JSON 파일들을 찾습니다
    json_files = []
    for folder in target_folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            json_files.extend(list(folder_path.glob("*.json")))
    
    print(f"총 {len(json_files)}개의 JSON 파일을 처리합니다...")
    
    # tqdm으로 진행 상황을 표시하며 파일들을 처리
    for file_path in tqdm(json_files):
        process_json_file(file_path)
        print(f"\n처리 완료: {file_path}")

if __name__ == "__main__":
    main() 