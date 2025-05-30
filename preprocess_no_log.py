## plumber와 pymupdf로 추출한 JSON 파일을 병합하고 전처리하는 스크립트

## from preprocess import * 후 merge_for_plumber(pdfplumber_path) 또는 merge_for_mup(pymupdf_path)로 호출하여 사용

import json
import re
import os
import unicodedata
import glob
from collections import OrderedDict

## 수정 필요
pdfplumber_path = "C:/Users/user/OneDrive/Deesktop/mid_project/output/pdfplumber_json"
pymupdf_path = "C:/Users/user/OneDrive/Deesktop/mid_project/output/pymupdf_json"


# CSV 파일 및 JSON 로딩 함수
def load_data(folder_path):
    full_data = []
    filenames = []  # 파일 이름을 저장할 리스트
    
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        return full_data, filenames

    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        full_data.append(data)
        filenames.append(os.path.basename(path))  # 경로에서 파일 이름만 추출하여 저장


    return full_data, filenames

# 노이즈 제거 및 문장 구조 정리
def remove_duplicate_lines(text):
    lines = text.split("\n")  # 텍스트를 줄 단위로 분리
    # 줄 끝 공백 제거 후 중복 제거: OrderedDict를 사용해 입력 순서 유지하며 중복 제거
    unique_lines = list(OrderedDict.fromkeys([line.strip() for line in lines if line.strip()]))
    return "\n".join(unique_lines)  # 다시 줄바꿈 문자로 연결해 하나의 문자열로 반환

# 문장 세부 노이즈 제거 
def clean_noise(text):
    # 1. 특정 표 패턴 제거: [[표]] ~ [[/표]] 사이에 "M: 1", "M: 2", ..., "M: 24" 포함 시 삭제  <- 위 결과에서 불필요하게 들어간 달력 텍스트를 지워버리기 위함
    def remove_specific_table(match):
        content = match.group(0)
        if re.search(r"M:\s*1.*M:\s*24", content, re.DOTALL) and "구분(월)" in content and "시범운영" in content:
            return ""  # 해당 패턴이 명확히 포함된 경우만 삭제
        return content  # 그렇지 않으면 원래 표 내용 유지

    text = re.sub(r"\[\[표\]\].*?\[\[/표\]\]", remove_specific_table, text, flags=re.DOTALL)

    # 2. 페이지 번호 또는 섹션 번호처럼 보이는 "- 숫자 -" 패턴 제거
    text = re.sub(r"-\s*\d+\s*-", "", text)

    # 3. 빈 괄호 "( )" 제거
    text = re.sub(r"\( *\)", "", text)

    # 4. "(RFP:)", "(RFP:: )" 등 RFP 관련 태그 패턴 제거
    text = re.sub(r"\(RFP\s*[:]*[:\s]*\)", "", text)

    # 5. ":::", "::::" 등 연속된 콜론을 ":" 하나로 정리
    text = re.sub(r"[:]{2,}", ":", text)

    # 6. "···", "•••" 등의 점 형태 불릿 마커 제거
    text = re.sub(r"[·•]{3,}", "", text)

    # 7. "....", ".." 등 반복된 마침표 제거
    text = re.sub(r"\.{2,}", "", text)

    # 8. 여러 개의 연속 공백을 하나로 축소
    text = re.sub(r"\s{2,}", " ", text)

    # 9. 여러 줄바꿈(\n\n\n 등)을 하나로 줄임
    text = re.sub(r"\n{2,}", "\n", text)

    # 10. 여전히 남아 있는 두 줄바꿈은 공백 하나로 대체 (문단 간격 제거)
    text = re.sub(r"\n\n", " ", text)

    return text

# 제목과 본문 병합
def merge_titles_and_paragraphs(text):
    lines = text.split("\n")  # 줄 단위로 텍스트 분리
    merged = []  # 병합된 문단들을 저장할 리스트
    buffer = ""  # 현재 병합 중인 문단 임시 저장

    for line in lines:
        # "1.", "1.2", "2.1.3" 등 번호로 시작하는 경우를 제목으로 인식
        if re.match(r"^\d+(\.\d+)*\s", line):
            if buffer:
                merged.append(buffer.strip())  # 기존 문단을 리스트에 추가
            buffer = line.strip()  # 새로운 제목으로 버퍼 초기화
        else:
            buffer += " " + line.strip()  # 제목이 아니라면 본문으로 간주하고 버퍼에 이어붙임

    if buffer:
        merged.append(buffer.strip())  # 마지막 문단 처리

    return "\n\n".join(merged)  # 문단 사이를 빈 줄로 구분하여 반환

# 전체 텍스트 전처리 함수
def preprocess_text(text):
    return merge_titles_and_paragraphs(remove_duplicate_lines(clean_noise(text)))

# 페이지 단위 청킹 함수
def merge_page_content(full_docs, filenames, min_length=500, max_length=3000):
    merged_full_docs = []

    for filename, doc in zip(filenames, full_docs):
        merged_pages = []
        visited = [False] * len(doc)
        page_idx = 0

        while page_idx < len(doc):
            if visited[page_idx]:
                page_idx += 1
                continue

            types = set()
            merged_page_number = float("inf")  # 대표 페이지 번호 초기화
            merged_text_parts = []

            current_idx = page_idx
            temp_idx = current_idx

            while temp_idx < len(doc) and not visited[temp_idx]:
                page_texts = [line['text'] for line in doc[temp_idx] if 'text' in line]
                types.update(line['type'] for line in doc[temp_idx] if 'type' in line)
                page_text = preprocess_text("\n".join(page_texts))

                merged_text_parts.append(page_text)
                merged_page_number = min(merged_page_number, temp_idx + 1)  # 최소 페이지 번호 갱신

                current_merged_length = sum(len(p) for p in merged_text_parts)

                if current_merged_length >= min_length:
                    next_idx = temp_idx + 1
                    if next_idx < len(doc):
                        next_page_texts = [line['text'] for line in doc[next_idx] if 'text' in line]
                        next_page_text = preprocess_text("\n".join(next_page_texts))
                        next_length = len(next_page_text)
                        if next_length < min_length:
                            temp_idx += 1
                            continue
                        else:
                            break
                    else:
                        break
                else:
                    temp_idx += 1

            for i in range(page_idx, temp_idx + 1):
                visited[i] = True

            merged_text = "\n".join(merged_text_parts).strip()

            if len(merged_text) < min_length and merged_pages:
                # 너무 짧으면 이전 블록에 병합
                prev_block = merged_pages[-1]
                prev_block["merged_page_content"] += "\n" + merged_text
                prev_block["page_number"] = [min(prev_block["page_number"][0], merged_page_number)]
                prev_block["types"] = list(set(prev_block["types"]) | types)

            elif len(merged_text) > max_length:
                sub_lst = re.split(r'(\[\[/표\]\])', merged_text)
                reconstructed = []

                for i in range(0, len(sub_lst), 2):
                    sub = sub_lst[i].strip()
                    if i + 1 < len(sub_lst):
                        sub += sub_lst[i + 1]
                    if len(sub) > max_length:
                        sub_sub_lst = sub.split('|')
                        chunk_size = len(sub_sub_lst) // 3
                        chunks = [
                            '|'.join(sub_sub_lst[:chunk_size + 5]),
                            '|'.join(sub_sub_lst[chunk_size - 5:2 * chunk_size + 5]),
                            '|'.join(sub_sub_lst[2 * chunk_size + 5:])
                        ]
                        reconstructed.extend([chunk for chunk in chunks if chunk.strip()])
                    else:
                        reconstructed.append(sub)

                temp_buffer = ""
                for part in reconstructed:
                    part = part.strip()
                    if not part:
                        continue
                    if len(temp_buffer) + len(part) < max_length:
                        temp_buffer += "\n" + part
                    else:
                        if temp_buffer.strip():
                            merged_pages.append({
                                "filename": filename,
                                "page_number": int(merged_page_number),
                                "types": ",".join(list(types)),
                                "merged_page_content": temp_buffer.strip(),
                            })
                        temp_buffer = part

                if temp_buffer.strip():
                    merged_pages.append({
                        "filename": filename,
                        "page_number": int(merged_page_number),
                        "types": ",".join(list(types)),
                        "merged_page_content": temp_buffer.strip(),
                    })

            else:
                merged_pages.append({
                    "filename": filename,
                    "page_number": int(merged_page_number),
                    "types": ", ".join(list(types)),
                    "merged_page_content": merged_text,
                })

            page_idx = temp_idx + 1

        merged_full_docs.append(merged_pages)

    return merged_full_docs

# 최종 데이터 출력 함수 (for pymupdf)
def merge_for_mup(mup_path):
    print("=== PyMuPDF Data Processing Start ===")
    
    full_docs, filenames = load_data(mup_path)
    
    if not full_docs:
        print("No PyMuPDF data found. Skipping.")
        return
        
    merged_mup_docs = merge_page_content(full_docs, filenames)
    
    output_file = "merged_mup_data.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_mup_docs, f, ensure_ascii=False, indent=2)
        print(f"PyMuPDF data saved: {output_file}")
    except Exception as e:
        print(f"Failed to save PyMuPDF data: {e}")

# 최종 데이터 출력 함수 (for pdfplumber)
def merge_for_plumber(plumber_path):
    print("=== PDFPlumber Data Processing Start ===")
    
    full_docs, filenames = load_data(plumber_path)
    
    if not full_docs:
        print("No PDFPlumber data found. Skipping.")
        return
        
    merged_plumber_docs = merge_page_content(full_docs, filenames)

    output_file = "merged_plumber_data.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_plumber_docs, f, ensure_ascii=False, indent=2)

        print(f"PDFPlumber data saved: {output_file}")
    except Exception as e:
        print(f"Failed to save PDFPlumber data: {e}")

def main():
    
    print("=== Data Preprocessing Started ===")
    

    # 전역 변수 사용
    global pymupdf_path, pdfplumber_path
    


    # 병합 및 전처리 실행
    try:
        merge_for_mup(pymupdf_path)
        merge_for_plumber(pdfplumber_path)
        
        print("=== All Preprocessing Completed Successfully ===")
        
    except Exception as e:
        error_msg = f"Error during preprocessing: {e}"
        print(error_msg)

# 인코딩 문제 해결
import sys
import os
import io

# Windows에서 UTF-8 출력을 위한 설정
if sys.platform.startswith('win'):
    # 환경변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # stdout을 UTF-8로 강제 설정
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')

# 스크립트 시작 메시지
print("=== Script Loaded Successfully ===")

if __name__ == "__main__":
    main()
else:
    print("Module imported successfully.")