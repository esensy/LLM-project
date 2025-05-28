# main.py

import json, os, sys, torch

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(ROOT_PATH)
if PARENT_DIR not in sys.path:
  sys.path.insert(0, PARENT_DIR)

from huggingface_hub import notebook_login
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.load import dumps, loads

from 제안요청서.load_embedding_model import load_embedding_model
from 제안요청서.config import (EMBEDDING_MODEL_NAME, METADATA_PATH, DOCS_PATH, 
                          CHROMA_METADATA_PATH, CHROMA_DOCUMENT_PATH, CHAT_MODEL_NAME, 
                          CHAT_TEMPERATURE, CHAT_N_RESPONSES, TOP_K_RETRIEVAL, 
                          PROJECT_ROOT, SAVE_PATH)
from 제안요청서.utils.utils import clean_filename, print_divider, load_metadata, load_documents_pre_chunked
from 제안요청서.database.database_utils import create_metadata_db, create_document_db 
from 제안요청서.query.query import process_query

# 메인 함수.
def main():
  # 1. CUDA 사용, Hugging Face 로그인, OpenAI API 키 설정.
  notebook_login()
  print(f"CUDA 사용 가능: {torch.cuda.is_available()}.")
  OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # OpenAI key 설정.
  
  if not OPENAI_API_KEY:
    OPENAI_API_KEY = input("OpenAI API 키를 입력하세요: ")
    print(f"Warning: OPENAI_API_KEY environment variable not found. Using manually entered key.")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

  # 2. 임베딩 모델 로드.
  print("🌟 임베딩 모델 로드 중....")
  embeddings = load_embedding_model(EMBEDDING_MODEL_NAME)

  # 3. 메타데이터 로드.
  print("\n🪄 메타데이터 로드 중....")
  metadata_df = load_metadata(METADATA_PATH)

  # 4. 문서 로드.
  print("\n🎆 문서 로드 중....")
  pre_chunked_data = load_documents_pre_chunked(DOCS_PATH)

  # 5. 메타데이터 벡터 데이터베이스 생성.
  print("\n✨ 메타데이터 벡터 데이터베이스 생성 중....")
  metadata_db = create_metadata_db(metadata_df, embeddings, CHROMA_METADATA_PATH)

  # 6. 문서 벡터 데이터베이스 생성.
  print("\n🎉 문서 벡터 데이터베이스 생성 중....")
  document_db, bm25_retriever = create_document_db(pre_chunked_data, metadata_df, embeddings, CHROMA_DOCUMENT_PATH)

  # 7. 대화형 인터페이스.
  print("\n=== 공공입찰 공고 문서 질의응답 시스템 시작 ===")
  print("질문을 입력하세요. 종료하려면 'exit'를 입력하세요.")

  # 8. ChatGPT 모델 설정.
  chat = ChatOpenAI(
      model = CHAT_MODEL_NAME,
      temperature = CHAT_TEMPERATURE,
      n = CHAT_N_RESPONSES,
      openai_api_key = os.environ["OPENAI_API_KEY"]
  )

  chat_history = [] # 대화 기록.
  all_results = []  # 결과 담을 리스트.
  while True:
    print_divider()
    query = input("\n질문: ")
    if query.lower() == "exit":
      break

    # 질문 처리 및 응답 생성.
    result = process_query(chat, query, metadata_db, document_db, bm25_retriever, embeddings, metadata_df, top_k = TOP_K_RETRIEVAL, re_rank = True, chat_history = chat_history)

    # 결과.
    all_results.append({query: result})

    # 대화 기록 업데이트.
    chat_history.extend([HumanMessage(content = query), AIMessage(content = result["response"])]) # "human" 및 "ai" 키워드임.

    # 적당히 기록.
    chat_history = chat_history[-6:]

    # 응답 출력.
    print("\n\033[91m=== 응답 ===\033[0m")
    print(result["response"])

    # 관련 파일 정보 출력.
    print("\n\033[93m=== 관련 파일 ===\033[0m")
    for file in result["relevant_files"]:
      print(f"- {file}")

    # 메인 함수에서 결과 출력 시 메타데이터 컨텍스트도 확인.
    print("\n\033[95m=== 메타데이터 컨텍스트 ===\033[0m")
    if "metadata_context" in result:
      metadata_context = result["metadata_context"]
      print(metadata_context)
    else:
      print("메타데이터 컨텍스트가 없습니다.")

    # 참조 청크 출력 코드 추가.
    print("\n\033[94m=== 참조한 청크 정보 ===\033[0m")
    if result["referenced_chunks"]:
      for i, chunk in enumerate(result["referenced_chunks"], 1):
        print(f"\n\033[92m--- 참조 청크 {i} ---\033[0m")
        print(f"사업명: {chunk['사업명']}")
        print(f"발주기관: {chunk['발주기관']}")
        print(f"유사도 점수: {chunk['similarity_score']:.4f}")

        # 청크 내용 요약 출력 (너무 길면 잘라서 출력).
        content = chunk["content"]
        if len(content) > 500:
          print(f"내용 미리보기: {content[:500]}....")
          print(f"(총 {len(content)}자).")
        else:
          print(f"내용: {content}.")
    else:
      print("참조한 청크가 없습니다.")

  # 파일에 저장하기.
  # 객체를 JSON 문자역로 먼저 변환.
  json_str = dumps(all_results)
  with open(SAVE_PATH, "w", encoding = "utf-8") as f:
    f.write(json_str)

  print(f"\n모든 쿼리 결과 {SAVE_PATH} 경로로 저장 완료.")

if __name__ == "__main__":
  main()
 