import os
import fitz  # PyMuPDF
import logging
import warnings
import time
import json

logging.getLogger("pdfplumber").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

def is_table_line(line):
    # 줄에 탭 2개 이상(표 가능성) 또는 콤마 2개 이상(표 가능성)
    return line.count('\t') >= 2 or line.count(',') >= 2

def table_lines_to_kv(table_lines):
    if not table_lines or len(table_lines) < 2:
        return ""
    # 헤더 추출
    header = [h.strip() for h in table_lines[0].replace('\t', ',').split(',')]
    lines = []
    for row in table_lines[1:]:
        vals = [v.strip() for v in row.replace('\t', ',').split(',')]
        # 길이 맞추기
        vals += [''] * (len(header) - len(vals))
        pairs = []
        for h, v in zip(header, vals):
            if h or v:
                pairs.append(f"{h}: {v}")
        if pairs:
            lines.append(" | ".join(pairs))
    return "\n".join(lines)

def extract_text_and_tables_pymupdf(pdf_path):
    merged_content = []
    json_by_page = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        lines = text.splitlines()
        page_content = []
        page_blocks = []
        table_mode = False
        table_block = []
        for line in lines:
            if is_table_line(line):
                if not table_mode:
                    table_mode = True
                    table_block = []
                table_block.append(line)
            else:
                if table_mode and table_block:
                    kv_table = table_lines_to_kv(table_block)
                    if kv_table.strip():
                        # 텍스트(문자열) 리스트
                        page_content.append("[[표]]\n" + kv_table + "\n[[/표]]")
                        # JSON(딕셔너리) 리스트
                        page_blocks.append({"type": "table", "text": kv_table})
                    table_mode = False
                    table_block = []
                if line.strip():
                    page_content.append(line)
                    page_blocks.append({"type": "text", "text": line})
        # 마지막 표 블록이 남아있는 경우
        if table_mode and table_block:
            kv_table = table_lines_to_kv(table_block)
            if kv_table.strip():
                page_content.append("[[표]]\n" + kv_table + "\n[[/표]]")
                page_blocks.append({"type": "table", "text": kv_table})
        merged_content.append(f"--- Page {page_num+1} ---\n" + "\n".join(page_content))
        json_by_page.append(page_blocks)
    doc.close()
    return "\n\n".join(merged_content), json_by_page

def extract_pdfs_with_pymupdf(input_dir, output_dir, json_dir):
    """
    Extract text and tables from PDF files using PyMuPDF.
    
    Args:
        input_dir (str): Directory containing PDF files
        output_dir (str): Directory to save text files
        json_dir (str): Directory to save JSON files
    """
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # 실제 실행
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            out_name = os.path.splitext(filename)[0]
            out_txt_path = os.path.join(output_dir, out_name + ".txt")
            out_json_path = os.path.join(json_dir, out_name + ".json")
            merged_content, json_content = extract_text_and_tables_pymupdf(pdf_path)
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(merged_content)
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
        # break # 디버깅
        
    print("PyMuPDF: Key-Value 변환, 페이지별 json 저장 완료!")
    print(f"{time.time()-start_time:.2f}초 소요")