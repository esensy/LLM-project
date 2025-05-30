## 사용법

### 1. RAG 시스템 실행

#### 기본 실행 (사전 정의된 질문 사용)
```bash
python main.py
```
또는
```bash
python main.py --mode predefined
```

#### 대화형 모드 실행
```bash
python main.py --mode interactive
```

#### 사용자 정의 질문으로 실행
```bash
python main.py --mode predefined --questions "첫 번째 질문" "두 번째 질문" "세 번째 질문"
```

**실행 모드 설명:**
- **predefined**: 사전에 정의된 질문들을 순차적으로 처리 (기본값)
- **interactive**: 사용자가 직접 질문을 입력하여 대화형으로 진행

**명령행 옵션:**
- `--mode`: 실행 모드 선택 (`predefined` 또는 `interactive`)
- `--questions`: 사용자 정의 질문 리스트 (predefined 모드에서만 사용)

**사용 예시:**
```bash
# 기본 모드 (사전 정의된 질문 사용)
python main.py

# 대화형 모드
python main.py --mode interactive

# 특정 질문들만 테스트
python main.py --mode predefined --questions "공고 번호가 20240821893인 사업 이름이 뭐야?" "사업 금액이 가장 큰 사업은 뭐야?"

# 단일 질문 테스트
python main.py --mode predefined --questions "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 알려줘."
```# RAG Project - 공공입찰 공고 문서 질의응답 시스템

LLM 호출을 최적화한 공공입찰 공고 문서 질의응답 시스템입니다.

## 주요 기능

1. **LLM 호출 최적화**
2. **처리 순서 개선**: 규칙 기반 → 발주기관 하이브리드 → LLM 기반
3. **대화 맥락 파악** 및 메타데이터 처리 통합
4. **처리 방법에 따른 명확한 구분**
5. **완전한 문서 처리 파이프라인**: HWP → PDF → 텍스트 추출 → 전처리

## 프로젝트 구조

```
rag_project/
├── .env                           # 환경변수 파일
├── .gitignore                     # Git 무시 파일
├── requirements.txt               # 의존성 패키지
├── main.py                       # RAG 시스템 메인 실행 파일
├── document_pipeline_main.py     # 문서 처리 파이프라인 메인 파일
├── config/
│   └── settings.py               # 설정 관리
├── models/
│   └── embeddings.py             # 임베딩 모델 관리
├── data/
│   ├── __init__.py
│   ├── loader.py                 # 데이터 로드 기능
│   └── preprocessor.py           # 데이터 전처리 기능
├── database/
│   ├── __init__.py
│   ├── vector_db.py              # 벡터 DB 생성 및 관리
│   └── metadata_db.py            # 메타데이터 DB 관리
├── search/
│   ├── __init__.py
│   ├── metadata_search.py        # 메타데이터 검색
│   ├── hybrid_search.py          # 하이브리드 검색
│   └── document_search.py        # 문서 검색
├── conversation/
│   ├── __init__.py
│   ├── manager.py                # 대화 맥락 관리
│   └── llm_processor.py          # LLM 기반 처리
├── utils/
│   ├── __init__.py
│   ├── text_utils.py             # 텍스트 처리 유틸리티
│   └── display_utils.py          # 출력 관련 유틸리티
├── response/
│   ├── __init__.py
│   └── generator.py              # 응답 생성
├── document_processing/          # 문서 처리 모듈
│   ├── __init__.py
│   ├── hwp_converter.py          # HWP → PDF 변환
│   ├── pdf_extractors/
│   │   ├── __init__.py
│   │   ├── pdfplumber_extractor.py    # PDFPlumber 텍스트 추출
│   │   └── pymupdf_extractor.py       # PyMuPDF 텍스트 추출
│   ├── text_preprocessor.py      # 텍스트 전처리
│   └── pipeline.py               # 전체 문서 처리 파이프라인
└── scripts/                      # 독립 실행 스크립트
    ├── convert_hwp_to_pdf.py     # HWP 변환 스크립트
    ├── extract_with_pdfplumber.py    # PDFPlumber 추출 스크립트
    ├── extract_with_pymupdf.py       # PyMuPDF 추출 스크립트
    └── preprocess_documents.py       # 문서 전처리 스크립트
```

## 설치 및 실행

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **환경변수 설정**
`.env` 파일을 생성하고 다음 내용을 추가:

```env
# === 필수 설정 ===
# OpenAI API 키
OPENAI_API_KEY=your_openai_api_key_here

# RAG 시스템 파일 경로 (반드시 설정 필요)
MUP_FILE_PATH=data/merged_mup_data.json
PLUMBER_FILE_PATH=data/merged_plumber_data.json
METADATA_PATH=data/data_list.csv

# === 선택적 설정 ===
# GPU 메모리 최적화 (GPU 사용 시 권장)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 문서 처리 파이프라인 경로 (문서 처리 기능 사용 시만 필요)
HWP_FILES_DIR=files
PDF_OUTPUT_DIR=pdf_files
PDF_INPUT_DIR=pdf_input
PDFPLUMBER_OUTPUT_DIR=output/pdfplumber
PDFPLUMBER_JSON_DIR=output/pdfplumber_json
PYMUPDF_OUTPUT_DIR=output/pymupdf
PYMUPDF_JSON_DIR=output/pymupdf_json

# 전처리 경로 (JSON 전처리 기능 사용 시만 필요)
PDFPLUMBER_PATH=output/pdfplumber_json
PYMUPDF_PATH=output/pymupdf_json
```

**중요**: `.env` 파일을 참고하여 본인 환경에 맞게 경로를 설정하세요.

## 사용법

### 1. RAG 시스템 실행

#### 기본 실행 (사전 정의된 질문 사용)
```bash
python main.py
```
또는
```bash
python main.py --mode predefined
```

#### 대화형 모드 실행
```bash
python main.py --mode interactive
```

#### 사용자 정의 질문으로 실행
```bash
python main.py --mode predefined --questions "첫 번째 질문" "두 번째 질문" "세 번째 질문"
```

**실행 모드 설명:**
- **predefined**: 사전에 정의된 질문들을 순차적으로 처리 (기본값)
- **interactive**: 사용자가 직접 질문을 입력하여 대화형으로 진행

**명령행 옵션:**
- `--mode`: 실행 모드 선택 (`predefined` 또는 `interactive`)
- `--questions`: 사용자 정의 질문 리스트 (predefined 모드에서만 사용)

**사용 예시:**
```bash
# 기본 모드 (사전 정의된 질문 사용)
python main.py

# 대화형 모드
python main.py --mode interactive

# 특정 질문들만 테스트
python main.py --mode predefined --questions "공고 번호가 20240821893인 사업 이름이 뭐야?" "사업 금액이 가장 큰 사업은 뭐야?"

# 단일 질문 테스트
python main.py --mode predefined --questions "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 알려줘."
```

### 2. 문서 처리 파이프라인 실행

**전체 파이프라인 (HWP → PDF → 텍스트 추출 → 전처리)**
```bash
python document_pipeline_main.py --mode full
```

**텍스트 추출만 실행**
```bash
python document_pipeline_main.py --mode extract
```

**전처리만 실행**
```bash
python document_pipeline_main.py --mode preprocess
```

### 3. 개별 스크립트 실행

**HWP를 PDF로 변환**
```bash
python scripts/convert_hwp_to_pdf.py --hwp-dir files --pdf-dir pdf_files
```

**PDFPlumber로 텍스트 추출**
```bash
python scripts/extract_with_pdfplumber.py --input-dir pdf_files --output-dir output/pdfplumber --json-dir output/pdfplumber_json
```

**PyMuPDF로 텍스트 추출**
```bash
python scripts/extract_with_pymupdf.py --input-dir pdf_files --output-dir output/pymupdf --json-dir output/pymupdf_json
```

**텍스트 전처리**
```bash
python scripts/preprocess_documents.py --pymupdf-path output/pymupdf_json --pdfplumber-path output/pdfplumber_json
```

## 모듈 설명

### config/settings.py
- 환경변수 관리
- API 키 설정
- 파일 경로 설정
- 모델 및 검색 파라미터 설정

### models/embeddings.py
- 한국어 특화 임베딩 모델 로드
- HuggingFace SBERT 모델 사용

### data/
- **loader.py**: JSON, CSV 데이터 로드
- **preprocessor.py**: 메타데이터 전처리 (날짜, 금액 변환)

### database/
- **vector_db.py**: 문서 벡터 DB 생성
- **metadata_db.py**: 메타데이터 벡터 DB 생성

### search/
- **metadata_search.py**: 규칙 기반 메타데이터 검색
- **hybrid_search.py**: 발주기관 + 내용 하이브리드 검색
- **document_search.py**: 메인 검색 처리 로직

### conversation/
- **manager.py**: 대화 맥락 관리 클래스
- **llm_processor.py**: LLM 기반 질문 분석 및 처리

### response/generator.py
- 컨텍스트 기반 응답 생성

### utils/
- **text_utils.py**: 텍스트 정리 유틸리티
- **display_utils.py**: 출력 포맷팅 유틸리티

### document_processing/
- **hwp_converter.py**: HWP 파일을 PDF로 변환
- **pdf_extractors/**: PDFPlumber와 PyMuPDF를 사용한 텍스트 추출
- **text_preprocessor.py**: 추출된 텍스트의 노이즈 제거 및 전처리
- **pipeline.py**: 전체 문서 처리 워크플로우 관리

### scripts/
독립 실행 가능한 스크립트들로, 각 단계를 개별적으로 실행할 수 있습니다.

## 워크플로우

1. **문서 변환**: HWP 파일을 PDF로 변환 (선택사항)
2. **텍스트 추출**: PDFPlumber와 PyMuPDF를 사용하여 PDF에서 텍스트와 표 추출
3. **전처리**: 추출된 텍스트의 노이즈 제거, 중복 제거, 청킹
4. **벡터 DB 구축**: 전처리된 텍스트로 임베딩 생성 및 벡터 DB 구축
5. **질의응답**: 사용자 질문에 대한 지능형 검색 및 응답 생성

## 주요 개선사항

1. **완전한 모듈화**: 기능별로 코드를 분리하여 유지보수성 향상
2. **환경변수 관리**: `.env` 파일로 민감한 정보 관리
3. **설정 중앙화**: `config/settings.py`로 모든 설정 통합
4. **원본 코드 보존**: 모든 로직과 프롬프트가 원본 그대로 유지
5. **문서 처리 파이프라인**: HWP부터 최종 RAG 시스템까지 완전 자동화
6. **독립 실행 스크립트**: 각 단계를 개별적으로 실행 가능
7. **Git 준비**: `.gitignore` 파일로 불필요한 파일 제외

## 주의사항

- **Windows 환경**: HWP 변환 기능은 Windows와 한글과컴퓨터 한/글이 설치된 환경에서만 작동합니다.
- **API 키**: OpenAI API 키가 필요합니다.
- **CUDA**: GPU 가속을 위해서는 CUDA 설치가 권장됩니다.

## 라이센스

이 프로젝트는 원본 코드의 모든 로직과 주석을 보존하면서 모듈화한 버전입니다.# RAG Project - 공공입찰 공고 문서 질의응답 시스템

LLM 호출을 최적화한 공공입찰 공고 문서 질의응답 시스템입니다.

## 주요 기능

1. **LLM 호출 최적화** (최대 2번으로 제한)
2. **처리 순서 개선**: 규칙 기반 → 발주기관 하이브리드 → LLM 기반
3. **대화 맥락 파악** 및 메타데이터 처리 통합
4. **처리 방법에 따른 명확한 구분**

## 프로젝트 구조

```
rag_project/
├── .env                           # 환경변수 파일
├── .gitignore                     # Git 무시 파일
├── requirements.txt               # 의존성 패키지
├── main.py                       # 메인 실행 파일
├── config/
│   └── settings.py               # 설정 관리
├── models/
│   └── embeddings.py             # 임베딩 모델 관리
├── data/
│   ├── __init__.py
│   ├── loader.py                 # 데이터 로드 기능
│   └── preprocessor.py           # 데이터 전처리 기능
├── database/
│   ├── __init__.py
│   ├── vector_db.py              # 벡터 DB 생성 및 관리
│   └── metadata_db.py            # 메타데이터 DB 관리
├── search/
│   ├── __init__.py
│   ├── metadata_search.py        # 메타데이터 검색
│   ├── hybrid_search.py          # 하이브리드 검색
│   └── document_search.py        # 문서 검색
├── conversation/
│   ├── __init__.py
│   ├── manager.py                # 대화 맥락 관리
│   └── llm_processor.py          # LLM 기반 처리
├── utils/
│   ├── __init__.py
│   ├── text_utils.py             # 텍스트 처리 유틸리티
│   └── display_utils.py          # 출력 관련 유틸리티
└── response/
    ├── __init__.py
    └── generator.py              # 응답 생성
```

## 설치 및 실행

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **환경변수 설정**
`.env` 파일을 생성하고 다음 내용을 추가:
```
OPENAI_API_KEY=your_openai_api_key_here
MUP_FILE_PATH=path/to/merged_mup_data.json
PLUMBER_FILE_PATH=path/to/merged_plumber_data.json
METADATA_PATH=path/to/data_list.csv
```

3. **실행**
```bash
python main.py
```

## 모듈 설명

### config/settings.py
- 환경변수 관리
- API 키 설정
- 파일 경로 설정
- 모델 및 검색 파라미터 설정

### models/embeddings.py
- 한국어 특화 임베딩 모델 로드
- HuggingFace SBERT 모델 사용

### data/
- **loader.py**: JSON, CSV 데이터 로드
- **preprocessor.py**: 메타데이터 전처리 (날짜, 금액 변환)

### database/
- **vector_db.py**: 문서 벡터 DB 생성
- **metadata_db.py**: 메타데이터 벡터 DB 생성

### search/
- **metadata_search.py**: 규칙 기반 메타데이터 검색
- **hybrid_search.py**: 발주기관 + 내용 하이브리드 검색
- **document_search.py**: 메인 검색 처리 로직

### conversation/
- **manager.py**: 대화 맥락 관리 클래스
- **llm_processor.py**: LLM 기반 질문 분석 및 처리

### response/generator.py
- 컨텍스트 기반 응답 생성

### utils/
- **text_utils.py**: 텍스트 정리 유틸리티
- **display_utils.py**: 출력 포맷팅 유틸리티

## 주요 개선사항

1. **모듈화**: 기능별로 코드를 분리하여 유지보수성 향상
2. **환경변수 관리**: `.env` 파일로 민감한 정보 관리
3. **설정 중앙화**: `config/settings.py`로 모든 설정 통합
4. **원본 코드 보존**: 모든 로직과 프롬프트가 원본 그대로 유지
5. **Git 준비**: `.gitignore` 파일로 불필요한 파일 제외

## 사용법

시스템을 실행하면 미리 정의된 질문들에 대해 순차적으로 처리합니다. 
대화형 모드로 변경하려면 `main.py`의 주석 처리된 `while True` 루프를 활성화하세요.

## 라이센스

이 프로젝트는 원본 코드의 모든 로직과 주석을 보존하면서 모듈화한 버전입니다.