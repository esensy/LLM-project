"""
문서 처리 파이프라인 메인 실행 파일
HWP 변환, PDF 텍스트 추출, 전처리를 순차적으로 실행
"""

import argparse
from document_processing import (
    run_full_document_pipeline,
    run_extraction_only_pipeline,
    run_preprocessing_only_pipeline
)

def main():
    parser = argparse.ArgumentParser(description='Document Processing Pipeline')
    parser.add_argument('--mode', choices=['full', 'extract', 'preprocess'], default='full',
                       help='Pipeline mode: full (complete pipeline), extract (PDF extraction only), preprocess (preprocessing only)')
    parser.add_argument('--hwp-dir', help='Directory containing HWP files')
    parser.add_argument('--pdf-output-dir', help='Directory to save converted PDFs')
    parser.add_argument('--pdf-input-dir', help='Directory containing PDF files for extraction')
    parser.add_argument('--pdfplumber-output-dir', help='Output directory for PDFPlumber text files')
    parser.add_argument('--pdfplumber-json-dir', help='Output directory for PDFPlumber JSON files')
    parser.add_argument('--pymupdf-output-dir', help='Output directory for PyMuPDF text files')
    parser.add_argument('--pymupdf-json-dir', help='Output directory for PyMuPDF JSON files')
    parser.add_argument('--pdfplumber-path', help='Path to PDFPlumber JSON files for preprocessing')
    parser.add_argument('--pymupdf-path', help='Path to PyMuPDF JSON files for preprocessing')
    
    args = parser.parse_args()
    
    # 인수에서 None이 아닌 값들만 딕셔너리로 구성
    kwargs = {k: v for k, v in vars(args).items() if v is not None and k != 'mode'}
    
    print(f"=== Running Document Processing Pipeline (Mode: {args.mode}) ===")
    
    try:
        if args.mode == 'full':
            run_full_document_pipeline(**kwargs)
        elif args.mode == 'extract':
            run_extraction_only_pipeline(**kwargs)
        elif args.mode == 'preprocess':
            run_preprocessing_only_pipeline(**kwargs)
        
        print("=== Pipeline completed successfully ===")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())