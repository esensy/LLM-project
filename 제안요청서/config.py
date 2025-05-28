# config.py

import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CONFIG_DIR

# 데이터.
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCS_PATH = os.path.join(DATA_DIR, "merged_plumber_data.json")
METADATA_PATH = os.path.join(DATA_DIR, "data.csv")

# Chroma 데이터베이스.
CHROMA_METADATA_PATH = os.path.join(PROJECT_ROOT, "chroma_metadata")
CHROMA_DOCUMENT_PATH = os.path.join(PROJECT_ROOT, "chroma_document")

# 임베딩 및 re-ranker.
EMBEDDING_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
RERANKER_MODEL_NAME = "dragonkue/bge-reranker-v2-m3-ko"

# LLM.
CHAT_MODEL_NAME = "gpt-4.1-mini"
CHAT_TEMPERATURE = 0.1
CHAT_N_RESPONSES = 1 # Or 2, as you had before, but typically 1 for chat interaction
TOP_K_RETRIEVAL = 5

# 저장 경로.
SAVE_PATH = os.path.join(PROJECT_ROOT, "query_results.json")