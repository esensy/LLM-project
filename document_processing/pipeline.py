"""
Document processing pipeline that orchestrates the entire document processing workflow.
"""

from .hwp_converter import convert_all_hwp_to_pdf
from .pdf_extractors import extract_pdfs_with_pdfplumber, extract_pdfs_with_pymupdf
from .text_preprocessor import run_preprocessing
from config.settings import (
    HWP_FILES_DIR, PDF_OUTPUT_DIR, PDF_INPUT_DIR,
    PDFPLUMBER_OUTPUT_DIR, PDFPLUMBER_JSON_DIR,
    PYMUPDF_OUTPUT_DIR, PYMUPDF_JSON_DIR,
    PDFPLUMBER_PATH, PYMUPDF_PATH
)

def run_full_document_pipeline(
    hwp_dir=None,
    pdf_output_dir=None,
    pdf_input_dir=None,
    pdfplumber_output_dir=None,
    pdfplumber_json_dir=None,
    pymupdf_output_dir=None,
    pymupdf_json_dir=None,
    pdfplumber_path=None,
    pymupdf_path=None
):
    """
    Run the complete document processing pipeline:
    1. Convert HWP files to PDF (optional)
    2. Extract text using PDFPlumber
    3. Extract text using PyMuPDF
    4. Preprocess and merge the extracted data
    
    Args:
        hwp_dir (str, optional): Directory containing HWP files
        pdf_output_dir (str, optional): Directory to save converted PDFs
        pdf_input_dir (str, optional): Directory containing PDF files for extraction
        pdfplumber_output_dir (str, optional): Output directory for PDFPlumber text files
        pdfplumber_json_dir (str, optional): Output directory for PDFPlumber JSON files
        pymupdf_output_dir (str, optional): Output directory for PyMuPDF text files
        pymupdf_json_dir (str, optional): Output directory for PyMuPDF JSON files
        pdfplumber_path (str, optional): Path to PDFPlumber JSON files for preprocessing
        pymupdf_path (str, optional): Path to PyMuPDF JSON files for preprocessing
    """
    
    # Use default values from config if not provided
    hwp_dir = hwp_dir or HWP_FILES_DIR
    pdf_output_dir = pdf_output_dir or PDF_OUTPUT_DIR
    pdf_input_dir = pdf_input_dir or PDF_INPUT_DIR
    pdfplumber_output_dir = pdfplumber_output_dir or PDFPLUMBER_OUTPUT_DIR
    pdfplumber_json_dir = pdfplumber_json_dir or PDFPLUMBER_JSON_DIR
    pymupdf_output_dir = pymupdf_output_dir or PYMUPDF_OUTPUT_DIR
    pymupdf_json_dir = pymupdf_json_dir or PYMUPDF_JSON_DIR
    pdfplumber_path = pdfplumber_path or PDFPLUMBER_PATH
    pymupdf_path = pymupdf_path or PYMUPDF_PATH
    
    print("=== Starting Full Document Processing Pipeline ===")
    
    # Step 1: Convert HWP to PDF (optional)
    try:
        print("\nStep 1: Converting HWP files to PDF...")
        convert_all_hwp_to_pdf(hwp_dir, pdf_output_dir)
        print("HWP to PDF conversion completed.")
    except Exception as e:
        print(f"HWP conversion failed (this is optional): {e}")
    
    # Step 2: Extract text using PDFPlumber
    try:
        print("\nStep 2: Extracting text using PDFPlumber...")
        extract_pdfs_with_pdfplumber(pdf_input_dir, pdfplumber_output_dir, pdfplumber_json_dir)
        print("PDFPlumber extraction completed.")
    except Exception as e:
        print(f"PDFPlumber extraction failed: {e}")
        raise
    
    # Step 3: Extract text using PyMuPDF
    try:
        print("\nStep 3: Extracting text using PyMuPDF...")
        extract_pdfs_with_pymupdf(pdf_input_dir, pymupdf_output_dir, pymupdf_json_dir)
        print("PyMuPDF extraction completed.")
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        raise
    
    # Step 4: Preprocess and merge data
    try:
        print("\nStep 4: Preprocessing and merging extracted data...")
        run_preprocessing(pymupdf_path, pdfplumber_path)
        print("Preprocessing completed.")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise
    
    print("\n=== Full Document Processing Pipeline Completed Successfully ===")

def run_extraction_only_pipeline(
    pdf_input_dir=None,
    pdfplumber_output_dir=None,
    pdfplumber_json_dir=None,
    pymupdf_output_dir=None,
    pymupdf_json_dir=None
):
    """
    Run only the text extraction part of the pipeline (skip HWP conversion).
    
    Args:
        pdf_input_dir (str, optional): Directory containing PDF files for extraction
        pdfplumber_output_dir (str, optional): Output directory for PDFPlumber text files
        pdfplumber_json_dir (str, optional): Output directory for PDFPlumber JSON files
        pymupdf_output_dir (str, optional): Output directory for PyMuPDF text files
        pymupdf_json_dir (str, optional): Output directory for PyMuPDF JSON files
    """
    
    # Use default values from config if not provided
    pdf_input_dir = pdf_input_dir or PDF_INPUT_DIR
    pdfplumber_output_dir = pdfplumber_output_dir or PDFPLUMBER_OUTPUT_DIR
    pdfplumber_json_dir = pdfplumber_json_dir or PDFPLUMBER_JSON_DIR
    pymupdf_output_dir = pymupdf_output_dir or PYMUPDF_OUTPUT_DIR
    pymupdf_json_dir = pymupdf_json_dir or PYMUPDF_JSON_DIR
    
    print("=== Starting Text Extraction Pipeline ===")
    
    # Extract text using PDFPlumber
    try:
        print("\nExtracting text using PDFPlumber...")
        extract_pdfs_with_pdfplumber(pdf_input_dir, pdfplumber_output_dir, pdfplumber_json_dir)
        print("PDFPlumber extraction completed.")
    except Exception as e:
        print(f"PDFPlumber extraction failed: {e}")
        raise
    
    # Extract text using PyMuPDF
    try:
        print("\nExtracting text using PyMuPDF...")
        extract_pdfs_with_pymupdf(pdf_input_dir, pymupdf_output_dir, pymupdf_json_dir)
        print("PyMuPDF extraction completed.")
    except Exception as e:
        print(f"PyMuPDF extraction failed: {e}")
        raise
    
    print("\n=== Text Extraction Pipeline Completed Successfully ===")

def run_preprocessing_only_pipeline(pdfplumber_path=None, pymupdf_path=None):
    """
    Run only the preprocessing part of the pipeline.
    
    Args:
        pdfplumber_path (str, optional): Path to PDFPlumber JSON files for preprocessing
        pymupdf_path (str, optional): Path to PyMuPDF JSON files for preprocessing
    """
    
    # Use default values from config if not provided
    pdfplumber_path = pdfplumber_path or PDFPLUMBER_PATH
    pymupdf_path = pymupdf_path or PYMUPDF_PATH
    
    print("=== Starting Preprocessing Pipeline ===")
    
    try:
        run_preprocessing(pymupdf_path, pdfplumber_path)
        print("Preprocessing completed successfully.")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise
    
    print("=== Preprocessing Pipeline Completed Successfully ===")