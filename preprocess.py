## plumber와 pymupdf로 추출한 JSON 파일을 병합하고 전처리하는 스크립트

## from preprocess import * 후 merge_for_plumber(pdfplumber_path) 또는 merge_for_mup(pymupdf_path)로 호출하여 사용

import json
import re
import os
import unicodedata
import glob
import logging
from collections import OrderedDict

## 수정 필요
pdfplumber_path = "/content/drive/MyDrive/코드잇/PROJECT/2. 중급 프로젝트/huggingface/output/pdfplumber_json"
pymupdf_path = "/content/drive/MyDrive/코드잇/PROJECT/2. 중급 프로젝트/huggingface/output/pymupdf_json"

# CSV 파일 및 JSON 로딩 함수
def load_data(folder_path):
    full_data = []
    filenames = []  # 파일 이름을 저장할 리스트

    for path in glob.glob(os.path.join(folder_path, "*.json")):
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

# 페이지 단위 청킹 함수 (for pymupdf)
def merge_page_content_mup(full_docs, filenames, min_length=500, max_length=1500):
    merged_full_docs = []

    for filename, doc in zip(filenames, full_docs):
        merged_pages = []
        page_idx = 0

        while page_idx < len(doc):
            types = set()
            merged_text_parts = []
            merged_page_number = float("inf")

            current_length = 0
            start_idx = page_idx

            # --- 페이지 단위 병합 루프 ---
            while page_idx < len(doc) and current_length < max_length:
                page_texts = [line['text'] for line in doc[page_idx] if 'text' in line]
                page_text = preprocess_text("\n".join(page_texts))

                if not page_text:
                    page_idx += 1
                    continue

                types.update(line['type'] for line in doc[page_idx] if 'type' in line)
                merged_text_parts.append(page_text)
                merged_page_number = min(merged_page_number, page_idx + 1)
                current_length = sum(len(p) for p in merged_text_parts)

                if current_length >= max_length:
                    break

                page_idx += 1

            merged_text = "\n".join(merged_text_parts).strip()

            # --- 최대 길이 초과 시, [[표]] 청크 분할 & sub-표 태깅 ---
            if len(merged_text) > max_length:
                # 1) [[/표]] 기준 분리
                sub_lst = re.split(r'(\[\[/표\]\])', merged_text)
                reconstructed = []

                for i in range(0, len(sub_lst), 2):
                    sub = sub_lst[i].strip()
                    if i + 1 < len(sub_lst):
                        sub += sub_lst[i + 1]  # 닫는 태그 포함

                    if sub.startswith('[[표]]') and sub.endswith('[[/표]]'):
                        # 안의 내용만 추출
                        inner = sub[len('[[표]]'):-len('[[/표]]')].strip()

                        # 내용이 너무 긴 경우 분할
                        if len(sub) > max_length:
                            parts = inner.split('|')
                            temp_buffer = ""

                            for item in parts:
                                item = item.strip()
                                if not item:
                                    continue

                                chunk = (temp_buffer + '|' + item) if temp_buffer else item
                                if len(f"[[표]]{chunk}[[/표]]") <= max_length:
                                    temp_buffer = chunk
                                else:
                                    reconstructed.append(f"[[sub-표]][[표]]{temp_buffer.strip()}[[/표]][[/sub-표]]")
                                    temp_buffer = item

                            if temp_buffer:
                                reconstructed.append(f"[[sub-표]][[표]]{temp_buffer.strip()}[[/표]][[/sub-표]]")

                        else:
                            reconstructed.append(sub)

                    else:
                        # 일반 텍스트 처리
                        if len(sub) > max_length:
                            parts = sub.split('|')
                            temp_buffer = ""
                            for item in parts:
                                item = item.strip()
                                if not item:
                                    continue
                                chunk = (temp_buffer + '|' + item) if temp_buffer else item
                                if len(chunk) <= max_length:
                                    temp_buffer = chunk
                                else:
                                    reconstructed.append(f"[[sub-표]]{temp_buffer.strip()}[[/sub-표]]")
                                    temp_buffer = item
                            if temp_buffer:
                                reconstructed.append(f"[[sub-표]]{temp_buffer.strip()}[[/sub-표]]")
                        else:
                            reconstructed.append(f"[[sub-표]]{sub}[[/sub-표]]")


                # 3) 최종 청크 조합 및 페이지화
                temp_buffer = ""
                for part in reconstructed:
                    part = part.strip()
                    if not part:
                        continue

                    candidate = (temp_buffer + "\n" + part) if temp_buffer else part

                    # soft threshold: 1500 넘어도 허용
                    if len(candidate) <= max_length + 300:  # 1500보다 최대 300자 초과 허용
                        temp_buffer = candidate
                    else:
                        if len(temp_buffer) >= min_length:
                            merged_pages.append({
                                "filename": filename,
                                "page_number": int(merged_page_number),
                                "types": ", ".join(sorted(types)),
                                "merged_page_content": temp_buffer,
                            })
                        temp_buffer = part

                if len(temp_buffer) >= min_length:
                    merged_pages.append({
                        "filename": filename,
                        "page_number": int(merged_page_number),
                        "types": ", ".join(sorted(types)),
                        "merged_page_content": temp_buffer,
                    })

            # --- 정상 범위 내 페이지 추가 ---
            elif len(merged_text) >= min_length:
                merged_pages.append({
                    "filename": filename,
                    "page_number": int(merged_page_number),
                    "types": ", ".join(sorted(types)),
                    "merged_page_content": merged_text,
                })

            # --- 너무 짧으면 이전 블록에 병합 ---
            elif merged_pages:
                prev = merged_pages[-1]
                merged = prev["merged_page_content"] + "\n" + merged_text
                prev["merged_page_content"] = merged
                prev["page_number"] = min(prev["page_number"], merged_page_number)
                all_types = set(prev["types"].split(", ")) | types
                prev["types"] = ", ".join(sorted(all_types))

            page_idx += 1

        merged_full_docs.append(merged_pages)

    print(f"{len(merged_full_docs)}개의 문서를 병합했습니다.")
    return merged_full_docs

# 페이지 단위 청킹 함수 (for pdfplumber)
def merge_page_content_plumber(full_docs, filenames, min_length=500, max_length=3000):
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

    print(f"{len(merged_full_docs)}개의 문서를 병합했습니다.")
    return merged_full_docs

# 최종 데이터 출력 함수 (for pdfplumber)
def merge_for_plumber(plumber_path):
    full_docs, filenames = load_data(plumber_path)
    merged_plumber_docs = merge_page_content_plumber(full_docs, filenames)

    return merged_plumber_docs

# 최종 데이터 출력 함수 (for pymupdf)
def merge_for_mup(mup_path):
    full_docs, filenames = load_data(mup_path)
    merged_mup_docs = merge_page_content_mup(full_docs, filenames)

    return merged_mup_docs