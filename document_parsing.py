import os
from docling.document_converter import DocumentConverter
import re
from datetime import datetime
import json
import logging
import shutil
import tempfile

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cleanup_temp_files():
    """임시 파일 정리"""
    try:
        temp_dir = os.path.join(os.path.expanduser("~"), ".EasyOCR")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        logger.info("임시 파일 정리 완료")
    except Exception as e:
        logger.error(f"임시 파일 정리 중 오류 발생: {str(e)}")

def process_pdf_folder(pdf_folder="./PDF_1", output_folder="./processed_docs"):
    """PDF 폴더의 모든 파일을 처리하는 함수"""
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # PDF 파일 목록 가져오기
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning("처리할 PDF 파일이 없습니다.")
        return
    
    logger.info(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
    
    # 각 PDF 파일 처리
    for pdf_file in pdf_files:
        try:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            logger.info(f"\n처리 중: {pdf_file}")
            
            # 임시 디렉토리 생성
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # PDF를 마크다운으로 변환
                    converter = DocumentConverter()
                    result = converter.convert(pdf_path)
                    markdown_text = result.document.export_to_markdown()
                    
                    if not markdown_text.strip():
                        logger.warning(f"텍스트를 추출할 수 없습니다: {pdf_file}")
                        continue
                    
                    # 태깅 적용
                    tagged_text = title_tagging(markdown_text)
                    
                    # 청크로 분할
                    chunks = split_into_chunks(tagged_text)
                    
                    # 결과 저장
                    save_processed_document(pdf_file, chunks, output_folder)
                    
                    logger.info(f"처리 완료: {pdf_file}")
                    
                except Exception as e:
                    logger.error(f"PDF 처리 중 오류 발생 ({pdf_file}): {str(e)}")
                    continue
                
                finally:
                    # 임시 파일 정리
                    cleanup_temp_files()
                
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생 ({pdf_file}): {str(e)}")
            continue

def title_tagging(text):
    """텍스트에서 제목 태그를 찾아 HTML 태그로 변환"""
    lines = text.split('\n')
    tagged_lines = []
    
    for line in lines:
        # 마크다운 제목 (#) 처리
        if line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            tag = f'<h{level}>{line.lstrip("#").strip()}</h{level}>'
            tagged_lines.append(tag)
        # 한국어 문서 제목 패턴 처리
        else:
            title_pattern = r'^(제?\d+[장절]\.?|\d+\.\d+(\.\d+)?)\s+'
            match = re.match(title_pattern, line)
            if match:
                level = len(match.group(1).split('.'))
                tag = f'<h{min(level, 6)}>{line.strip()}</h{min(level, 6)}>'
                tagged_lines.append(tag)
            else:
                tagged_lines.append(line)
    
    return '\n'.join(tagged_lines)

def split_into_chunks(text, chunk_size=1000, overlap=200):
    """텍스트를 청크로 분할"""
    # 문단으로 분할
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def save_processed_document(filename, chunks, output_folder):
    """처리된 문서를 JSON 형식으로 저장"""
    output = {
        "filename": filename,
        "processed_date": datetime.now().isoformat(),
        "total_chunks": len(chunks),
        "chunks": []
    }
    
    for i, chunk in enumerate(chunks):
        output["chunks"].append({
            "chunk_index": i,
            "content": chunk
        })
    
    # 파일명에서 확장자 제거
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}_processed.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"저장 완료: {output_path}")

if __name__ == "__main__":
    try:
        process_pdf_folder()
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
    finally:
        cleanup_temp_files()