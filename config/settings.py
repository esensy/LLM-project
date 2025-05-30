import os
import torch
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# GPU 메모리 최적화 설정
if torch.cuda.is_available():
    # GTX 1060 3GB 메모리 최적화
    torch.cuda.empty_cache()
    # 메모리 할당 전략 설정
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

# RAG 시스템 파일 경로 설정 (필수 - .env에서 설정 필요)
MUP_FILE_PATH = os.getenv("MUP_FILE_PATH")
PLUMBER_FILE_PATH = os.getenv("PLUMBER_FILE_PATH")
METADATA_PATH = os.getenv("METADATA_PATH")

# 문서 처리 경로 설정 (선택적 - .env에서 설정 가능)
HWP_FILES_DIR = os.getenv("HWP_FILES_DIR")
PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR")
PDF_INPUT_DIR = os.getenv("PDF_INPUT_DIR")
PDFPLUMBER_OUTPUT_DIR = os.getenv("PDFPLUMBER_OUTPUT_DIR")
PDFPLUMBER_JSON_DIR = os.getenv("PDFPLUMBER_JSON_DIR")
PYMUPDF_OUTPUT_DIR = os.getenv("PYMUPDF_OUTPUT_DIR")
PYMUPDF_JSON_DIR = os.getenv("PYMUPDF_JSON_DIR")

# 전처리 경로 설정 (선택적 - .env에서 설정 가능)
PDFPLUMBER_PATH = os.getenv("PDFPLUMBER_PATH")
PYMUPDF_PATH = os.getenv("PYMUPDF_PATH")

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
# GPU 메모리가 부족할 경우 대안 모델
EMBEDDING_MODEL_NAME_FALLBACK = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM 모델 설정
LLM_MODEL_NAME = "gpt-4.1-mini"  # 더 가벼운 모델 사용
LLM_TEMPERATURE = 0
LLM_SEED = 42

# 검색 설정
DEFAULT_TOP_K = 5
METADATA_SIMILARITY_THRESHOLD = 1.5
DYNAMIC_SIMILARITY_THRESHOLD = 1.8

# 대화 맥락 설정
MAX_CONVERSATION_HISTORY = 5
MAX_RECENT_CONTEXT = 3

# 텍스트 전처리 설정
MIN_CHUNK_LENGTH = 500
MAX_CHUNK_LENGTH = 3000

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_MEMORY_LIMIT = 2.5  # GB, GTX 1060 3GB에서 안전한 사용량