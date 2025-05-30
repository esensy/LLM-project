"""
독립 실행 스크립트: HWP 파일을 PDF로 변환
"""

import os
import sys
import argparse

# 프로젝트 루트 디렉토리를 Python path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.hwp_converter import convert_all_hwp_to_pdf
from config.settings import HWP_FILES_DIR, PDF_OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description='Convert HWP files to PDF format')
    parser.add_argument('--hwp-dir', default=HWP_FILES_DIR, 
                       help=f'Directory containing HWP files (default: {HWP_FILES_DIR})')
    parser.add_argument('--pdf-dir', default=PDF_OUTPUT_DIR, 
                       help=f'Directory to save PDF files (default: {PDF_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    print("=== HWP to PDF Conversion Script ===")
    print(f"HWP files directory: {args.hwp_dir}")
    print(f"PDF output directory: {args.pdf_dir}")
    
    if not os.path.exists(args.hwp_dir):
        print(f"Error: HWP directory '{args.hwp_dir}' does not exist.")
        return 1
    
    try:
        convert_all_hwp_to_pdf(args.hwp_dir, args.pdf_dir)
        print("=== Conversion completed successfully ===")
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1

if __name__ == "__main__":
    exit(main())