import os
import sys
import contextlib
import pdfplumber
import time
import json

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def table_to_kv(table):
    if not table or len(table) < 2:
        return ""
    headers = [h.strip() if h else "" for h in table[0]]
    lines = []
    for row in table[1:]:
        vals = list(row) + [''] * (len(headers) - len(row))
        pairs = []
        for h, v in zip(headers, vals):
            h_clean = h.strip() if h else ""
            v_clean = (v or "").strip()
            if h_clean or v_clean:
                pairs.append(f"{h_clean}: {v_clean}")
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)

def extract_text_and_tables_no_table_dup(pdf_path):
    content_by_page = []      # txt 파일용
    json_by_page = []         # json 파일용
    with suppress_output():
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                blocks = []
                table_bboxes = []
                # 1. 표 추출(좌표 기억)
                for table in page.extract_tables():
                    words = page.extract_words()
                    cell_texts = [cell for row in table for cell in row if cell]
                    y0s, y1s = [], []
                    for word in words:
                        if word['text'] in cell_texts:
                            y0s.append(word['top'])
                            y1s.append(word['bottom'])
                    if y0s and y1s:
                        min_y, max_y = min(y0s), max(y1s)
                        table_bboxes.append((min_y - 1, max_y + 1))
                    table_text = table_to_kv(table)
                    if table_text.strip():
                        block = {'type': 'table', 'y0': y0s[0] if y0s else 0, 'text': f"[[표]]\n{table_text}\n[[/표]]"}
                        blocks.append(block)
                # 2. 텍스트 추출(표 영역 제외)
                for word in page.extract_words(use_text_flow=True, keep_blank_chars=True):
                    y = word['top']
                    in_table = False
                    for bbox in table_bboxes:
                        if bbox[0] <= y <= bbox[1]:
                            in_table = True
                            break
                    if not in_table:
                        block = {'type': 'text', 'y0': y, 'text': word['text']}
                        blocks.append(block)
                # 3. y좌표순 정렬 & 출력
                blocks.sort(key=lambda b: b["y0"])
                page_content = [b["text"] for b in blocks]
                content_by_page.append(f"--- Page {page_num} ---\n" + "\n".join(page_content))
                # blocks는 그대로 json에 저장
                json_by_page.append(blocks)
    return "\n\n".join(content_by_page), json_by_page

def extract_pdfs_with_pdfplumber(input_dir, output_dir, json_dir):
    """
    Extract text and tables from PDF files using pdfplumber.
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save text files
        json_dir (str): Directory to save JSON files
    """
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 실행
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            out_name = os.path.splitext(filename)[0]
            out_txt_path = os.path.join(output_dir, out_name + ".txt")
            out_json_path = os.path.join(json_dir, out_name + ".json")
            merged_content, json_content = extract_text_and_tables_no_table_dup(pdf_path)
            # txt 저장 (기존과 동일)
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(merged_content)
            # json 저장 (페이지별 구조, indent 예쁘게)
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
        # break # 디버깅

    print("pdfplumber 중복 없는 표 추출, txt & 페이지별 json 저장 완료!")
    print(f"{time.time()-start_time:.4f} 초 소요됨")