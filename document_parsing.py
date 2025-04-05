'''
!pip install docling

from docling.document_converter import DocumentConverter

source = "/content/2024년 3차 청년안심주택.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

with open('result.txt', 'w') as f:
    f.write(result.document.export_to_markdown())


import re

def title_tagging(text):
  """
  텍스트에서 '#'으로 시작하는 줄을 찾아 제목 태그로 변환합니다.

  Args:
    text: 입력 텍스트

  Returns:
    제목 태그가 적용된 텍스트
  """
  lines = text.split('\n')
  tagged_lines = []
  for line in lines:
    if line.startswith('#'):
      level = len(line) - len(line.lstrip('#'))  # '#' 개수로 제목 레벨 결정
      tag = f'<h{level}>{line.lstrip("#").strip()}</h{level}>'  # 제목 태그 생성
      tagged_lines.append(tag)
    else:
      tagged_lines.append(line)
  return '\n'.join(tagged_lines)

# result.txt 파일 읽기
with open('result.txt', 'r') as f:
  text = f.read()

# 제목 태깅 적용
tagged_text = title_tagging(text)

# 결과 출력 또는 파일 저장
print(tagged_text)
with open('tagged_result.txt', 'w') as f:
  f.write(tagged_text)
'''

import os

def split_into_chunks(text, chunk_size=1000):
  """
  텍스트를 청크 단위로 나눕니다.

  Args:
    text: 입력 텍스트
    chunk_size: 청크 크기 (기본값: 1000)

  Returns:
    청크 리스트
  """
  chunks = []
  for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)
  return chunks

# tagged_result.txt 파일 읽기
with open('tagged_result.txt', 'r') as f:
  tagged_text = f.read()

# 청크 단위로 나누기
chunks = split_into_chunks(tagged_text, chunk_size=2000)  # 청크 크기 조정 가능

# 폴더 생성
folder_name = 'chunks'  # 폴더 이름 지정
os.makedirs(folder_name, exist_ok=True)  # 폴더가 없으면 생성

# 청크별로 파일 저장
for i, chunk in enumerate(chunks):
  file_path = os.path.join(folder_name, f'chunk_{i}.txt')  # 파일 경로 생성
  with open(file_path, 'w') as f:
    f.write(chunk)

print(f'{len(chunks)}개의 청크 파일로 분할되어 {folder_name} 폴더에 저장되었습니다.')