"""
독립 실행 스크립트: PDFPlumber를 사용하여 PDF에서 텍스트 추출
"""

import os
import sys
import argparse

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.pdf_extractors import extract_pdfs_with_pdfplumber
from config.settings import PDF_INPUT_DIR, PDFPLUMBER_OUTPUT_DIR, PDFPLUMBER_JSON_DIR

def main():
    parser = argparse.ArgumentParser(description='Extract text and tables from PDF files using PDFPlumber')
    parser.add_argument('--input-dir', default=PDF_INPUT_DIR, 
                       help=f'Directory containing PDF files (default: {PDF_INPUT_DIR})')
    parser.add_argument('--output-dir', default=PDFPLUMBER_OUTPUT_DIR, 
                       help=f'Directory to save text files (default: {PDFPLUMBER_OUTPUT_DIR})')
    parser.add_argument('--json-dir', default=PDFPLUMBER_JSON_DIR, 
                       help=f'Directory to save JSON files (default: {PDFPLUMBER_JSON_DIR})')
    
    args = parser.parse_args()
    
    print("=== PDFPlumber Text Extraction Script ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Text output directory: {args.output_dir}")
    print(f"JSON output directory: {args.json_dir}")
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return 1
    
    try:
        extract_pdfs_with_pdfplumber(args.input_dir, args.output_dir, args.json_dir)
        print("=== PDFPlumber extraction completed successfully ===")
        return 0
    except Exception as e:
        print(f"Error during extraction: {e}")
        return 1

if __name__ == "__main__":
    exit(main())