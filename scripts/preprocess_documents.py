"""
독립 실행 스크립트: 추출된 JSON 파일들을 전처리하고 병합
"""

import os
import sys
import argparse

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.text_preprocessor import run_preprocessing
from config.settings import PYMUPDF_PATH, PDFPLUMBER_PATH

def main():
    parser = argparse.ArgumentParser(description='Preprocess and merge extracted JSON files')
    parser.add_argument('--pymupdf-path', default=PYMUPDF_PATH, 
                       help=f'Path to PyMuPDF JSON files (default: {PYMUPDF_PATH})')
    parser.add_argument('--pdfplumber-path', default=PDFPLUMBER_PATH, 
                       help=f'Path to PDFPlumber JSON files (default: {PDFPLUMBER_PATH})')
    
    args = parser.parse_args()
    
    print("=== Document Preprocessing Script ===")
    print(f"PyMuPDF JSON path: {args.pymupdf_path}")
    print(f"PDFPlumber JSON path: {args.pdfplumber_path}")
    
    if not os.path.exists(args.pymupdf_path):
        print(f"Warning: PyMuPDF path '{args.pymupdf_path}' does not exist.")
    
    if not os.path.exists(args.pdfplumber_path):
        print(f"Warning: PDFPlumber path '{args.pdfplumber_path}' does not exist.")
    
    if not os.path.exists(args.pymupdf_path) and not os.path.exists(args.pdfplumber_path):
        print("Error: Neither PyMuPDF nor PDFPlumber paths exist.")
        return 1
    
    try:
        run_preprocessing(args.pymupdf_path, args.pdfplumber_path)
        print("=== Preprocessing completed successfully ===")
        return 0
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return 1

if __name__ == "__main__":
    exit(main())