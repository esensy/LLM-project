# database_utils.py

import os
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Any, Tuple

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.retrievers import BM25Retriever # Import BM25Retriever 

from 제안요청서.utils.utils import clean_filename

# 메타데이터 임베딩 데이터베이스 생성 함수 개선.
def create_metadata_db(metadata_df: pd.DataFrame, embeddings: Embeddings, db_persist_directory: str):
  """
  메타데이터 기반 벡터 데이터베이스 생성 (사업명 + 발주기관 기반)

  Args:
    metadata_df: 메타데이터를 포함하는 Pandas DataFrame.
    embeddings: 사용할 임베딩 모델 객체 (LangChain Embeddings 타입).
    db_persist_directory: 데이터베이스가 저장될 경로.

  Returns:
    생성된 Chroma 벡터 데이터베이스 객체.
  """
  documents = []

  print("메타데이터 문서 객체 생성 중...")
  if metadata_df is None or metadata_df.empty:
    print("❗ 오류: 메타데이터 DataFrame이 비어있거나 유효하지 않습니다.")
    return None

  for idx, row in metadata_df.iterrows():
    # 사업명과 발주기관을 함께 사용하여 임베딩.
    project_name = row["사업명"] if "사업명" in row else ""
    agency = row["발주 기관"] if "발주 기관" in row else ""
    file_name = row["파일명"].replace(".pdf", "").strip() if "파일명" in row else ""

    # 풍부한 컨텍스트로 임베딩 - 사업명(2번 반복) + 발주기관 + 핵심 키워드.
    # 사업명을 2번 반복하여 가중치를 높임.
    enriched_content = f"{project_name} {project_name} {agency}"

    # 키워드 보강 - 프로젝트명에 시스템, 구축, 용역 등의 키워드가 있으면 추가.
    keywords = ["시스템", "사업", "구축", "개발", "용역", "플랫폼", "고도화", "기능개선"]
    for keyword in keywords:
      if keyword in project_name:
        enriched_content += f" {keyword}"

    # Document 객체 생성.
    doc = Document(
        page_content = enriched_content,    # 풍부한 컨텍스트로 변경.
        metadata = {
            "사업명": project_name,
            "공고번호": row["공고 번호"] if "공고 번호" in row else "",
            "공고차수": row["공고 차수"] if "공고 차수" in row else "",
            "발주기관": agency,"사업금액": row["사업 금액"] if "사업 금액" in row else "",
            "공개일자": row["공개 일자"] if "공개 일자" in row else "",
            "입찰시작일": row["입찰 참여 시작일"] if "입찰 참여 시작일" in row else "",
            "입찰마감일": row["입찰 참여 마감일"] if "입찰 참여 마감일" in row else "",
            "사업요약": row["사업 요약"] if "사업 요약" in row else "",
            "파일명": file_name
        }
    )
    documents.append(doc)
  
  print(f"Chroma 메타데이터 데이터베이스 생성 시작. 저장 경로: {db_persist_directory}")
  try:
    # Chroma 데이터베이스 생성.
    os.makedirs(db_persist_directory, exist_ok=True)
    metadata_db = Chroma.from_documents(
      documents = documents,
      embedding = embeddings,
      persist_directory = db_persist_directory
    )
    print(f"메타데이터 벡터 데이터베이스 생성 완료: {len(documents)}개의 문서.")
    return metadata_db
  except Exception as e:
    print(f"❗ 오류: 메타데이터 벡터 데이터베이스 생성 중 오류 발생: {e}")
    return None

# 문서 임베딩 데이터베이스 생성 함수에 청크 길이 통계 추가.
def create_document_db(pre_chunked_data: List[List[Dict[str, Any]]], metadata_df: pd.DataFrame, embeddings: Embeddings, db_persist_directory: str) -> Tuple[Chroma, BM25Retriever]:
  """
  문서 청크 기반 벡터 데이터베이스 생성 및 BM25 Retriever 생성

  Args:
    pre_chunked_data: 사전 청크된 문서 데이터를 포함하는 리스트의 리스트.
    metadata_df: 메타데이터를 포함하는 Pandas DataFrame (청크에 메타데이터를 추가하는 데 사용됨).
    embeddings: 사용할 임베딩 모델 객체 (LangChain Embeddings 타입).
    db_persist_directory: 문서 벡터 데이터베이스가 저장될 경로.

  Returns:
    생성된 Chroma 문서 벡터 데이터베이스 객체와 BM25 Retriever 객체의 튜플.
    처리할 청크가 없거나 오류 발생 시 (None, None) 반환.
  """
  all_chunks = []

  # 메타데이터를 파일명으로 인덱싱하여 빠르게 접근할 수 있도록 함.
  metadata_dict = {}
  if metadata_df is not None and not metadata_df.empty:
    try:
        for idx, row in metadata_df.iterrows():
          file_name = row["파일명"].replace(".pdf", "").strip() if "파일명" in row else ""
          metadata_dict[file_name] = row.to_dict()
    except Exception as e:
      print(f"❗ 오류: 메타데이터 딕셔너리 생성 중 오류 발생: {e}.")

  print("문서 청크 처리 및 Document 객체 생성 중....")
  # 각 사전 청크된 항목에 대해 Document 객체 생성.
  # pre_chunked_data의 형식이 리스트[닥셔너리]로 가정합니다.
  if isinstance(pre_chunked_data, list):
    for i, inner_list in enumerate(pre_chunked_data):
      if isinstance(inner_list, list):
        for j, chunk_data in enumerate(inner_list):
          try:
            # 사전 청크된 데이터에서 정보 추출.
            page_content = chunk_data.get("merged_page_content", "").strip()
            filename = clean_filename(chunk_data.get("filename", "").strip()) # 파일명 다시 클리닝하여 메타데이터 딕셔너리와 매칭.
            page_number = chunk_data.get("page_number", None) # 페이지 번호 추출.

            if not page_content or not filename:
              # 내용이나 파일명이 없는 청크는 건너뛰기.
              print(f"❗ 경고: 내용 또는 파일명이 없는 청크 발견 (외부 색인 {i} *** 내부 색인 {j}). 건너띄웁니다.")
              print(f"\t ❎ Page Content(내용) 출력: '{page_content}'")
              continue

            # 해당 파일의 메타데이터 가져오기.
            file_metadata = metadata_dict.get(filename, {})

            # Document 객체 생성.
            doc = Document(
                page_content = page_content,
                metadata = {
                    "filename": filename,
                    "chunk_id": f"{i}_{j}]", # 청크 데이터의 순서대로 아이디 부여.
                    "page_number": page_number,
                    "사업명": file_metadata.get("사업명", ""),
                    "발주기관": file_metadata.get("발주 기관", ""),
                    "공고번호": file_metadata.get("공고 번호", ""),
                    "공고차수": file_metadata.get("공고 차수", ""),
                    "사업금액": file_metadata.get("사업 금액", ""),
                    "공개일자": file_metadata.get("공개 일자", ""),
                    "입찰시작일": file_metadata.get("입찰 참여 시작일", ""),
                    "입찰마감일": file_metadata.get("입찰 참여 마감일", ""),
                    "사업요약": file_metadata.get("사업 요약", "")
                }
            )
            all_chunks.append(doc)

          except Exception as e:
            print(f"❗ 청크 처리 중 오류 발생 (외부 색인 Index {i}, 내부 색인 {j}): {e}. 해당 청크를 건너띄웁니다.")

      else:
        # 외부 색인이지만 내부 색인 아닌 경우.
        print(f"❗ 경고: 예상치 못한 데이터 형식 발견 (색인 {i}). 이 항목을 건너띄웁니다.")

  else:
    print("❗ 오류: pre_chunked_data 형식이 리스트가 아닙니다.")
    return None, None # 오류 발생 시 None 반환.

  print(f"총 {len(all_chunks)}개의 사전 청크된 항목을 처리했습니다.")

  # 청크 길이 통계 출력 (이전 코드와 동일).
  # 청크가 하나라도 있을 경우에만 통계 계산 및 출력.
  if all_chunks:
    chunk_lengths = [len(doc.page_content) for doc in all_chunks]
    print(f"청크 길이 분포 (총 {len(chunk_lengths)}개):")
    print(f"- 최소 길이: {min(chunk_lengths)}.")
    print(f"- 최대 길이: {max(chunk_lengths)}.")
    print(f"- 평균 길이: {np.mean(chunk_lengths):.2f}.")
    print(f"- 중간값: {np.median(chunk_lengths):.2f}.")

    # 청크 길이 히스토그램 출력 (구간별 분포).
    bins = [0, 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900]
    hist, bin_edges = np.histogram(chunk_lengths, bins=bins)
    for i in range(len(hist)):
      print(f"- {int(bin_edges[i])}-{int(bin_edges[i+1])} 자: {hist[i]}개 ({hist[i]/len(chunk_lengths)*100:.1f}%)")
  else:
    print("처리할 청크가 없습니다. 데이터베이스를 생성하지 않습니다.")
    return None, None # 처리할 청크가 없으면 None 반환.

  print(f"Chroma 문서 벡터 데이터베이스 생성 시작. 저장 경로: {db_persist_directory}")
  document_db = None
  try:
    # Ensure the directory exists
    os.makedirs(db_persist_directory, exist_ok = True)
    document_db = Chroma.from_documents(
      documents = all_chunks,
      embedding = embeddings,
      persist_directory = db_persist_directory 
    )
    print("Chroma 문서 벡터 데이터베이스 생성 완료.")
  except Exception as e:
    print(f"❗ 오류: Chroma 문서 벡터 데이터베이스 생성 중 오류 발생: {e}.")

  # BM25 Retriever 생성.
  print("BM25 Retriever 생성 중....")
  bm25_retriever = None
  try:
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    print("BM25 Retriever 생성 완료.")
  except Exception as e:
    print(f"❗ 오류: BM25 Retriever 생성 중 오류 발생: {e}.")
    bm25_retriever = None 

  # 데이터베이스 정보 확인.
  if document_db:
    print("Chroma DB 정보 확인 중...")
    try:
      ids = document_db.get()["ids"]
      print(f"총 아이디 개수: {len(ids)} *** 고유 아이디 개수: {len(set(ids))}.")
      duplicate_ids = [k for k, v in Counter(ids).items() if v > 1]
      if duplicate_ids:
        print(f"중복된 아이디: {duplicate_ids}.")
      print(f"메타데이터 손상 확인: {document_db._collection.metadata}.")
    except Exception as e:
      print(f"❗ Chroma DB 확인 중 오류 발생: {e}.")
      # 오류가 발생하더라도 함수를 중단하지 않고 진행.

  if document_db is None and bm25_retriever is None:
    print("데이터베이스 또는 Retriever를 생성하는 데 실패했습니다.")
    return None, None
  else:
    print(f"문서 벡터 데이터베이스 및 BM25 Retriever 생성 프로세스 완료: {len(all_chunks)}개의 청크.")
    return document_db, bm25_retriever

if __name__ == "__main__":
  print("Testing create_metadata_db function...")

  # Create a dummy DataFrame
  dummy_metadata_data = {
      "공고 번호": ["111", "222", "333"],
      "공고 차수": ["00", "01", "00"],
      "사업명": ["AI 시스템 구축 사업", "클라우드 전환 용역", "데이터 분석 플랫폼 개발"],
      "사업 금액": ["1억", "5천만", "2억"],
      "발주 기관": ["과학기술정보통신부", "행정안전부", "빅데이터청"],
      "공개 일자": ["2023-01-01", "2023-02-01", "2023-03-01"],
      "입찰 참여 시작일": ["2023-01-10", "2023-02-10", "2023-03-10"],
      "입찰 참여 마감일": ["2023-01-15", "2023-02-15", "2023-03-15"],
      "사업 요약": ["AI 시스템 구축.", "클라우드 전환.", "데이터 플랫폼 개발."],
      "파일형식": ["pdf", "hwp", "xlsx"],
      "파일명": ["사업1.pdf", "클라우드_사업.hwp", "데이터.xlsx"]
  }
  dummy_metadata_df = pd.DataFrame(dummy_metadata_data)

  # Create a dummy embeddings object
  # This requires a minimal LangChain Embeddings-like object
  # For a real test, you'd instantiate a real embeddings model,
  # but for a unit test, a mock might be sufficient.
  # Let's use a simple mock for demonstration
  from langchain.embeddings.fake import FakeEmbeddings
  dummy_embeddings = FakeEmbeddings(size = 10) # Specify a size for the fake embeddings

  # Define a test persist directory
  test_db_dir = "./test_chroma_metadata"
  # Clean up previous test runs
  if os.path.exists(test_db_dir):
    print(f"Cleaning up previous test directory: {test_db_dir}")
    import shutil
    shutil.rmtree(test_db_dir)

  # Create the database
  metadata_db = create_metadata_db(dummy_metadata_df, dummy_embeddings, test_db_dir)

  if metadata_db:
    print("\nMetadata database created successfully.")
    # You can perform some basic checks here, e.g., count documents
    print(f"Number of documents in DB: {metadata_db._collection.count()}")

  print("Testing create_document_db function...")

  dummy_pre_chunked_data = [
    [ # File 1
      {"merged_page_content": "AI 시스템 구축 사업의 개요", "filename": "사업1.pdf", "page_number": 1},
      {"merged_page_content": "요구사항 분석 및 설계", "filename": "사업1.pdf", "page_number": 5},
      {"merged_page_content": "구현 및 테스트", "filename": "사업1.pdf", "page_number": 10},
    ],
    [ # File 2
      {"merged_page_content": "클라우드 전환 목표 및 범위", "filename": "클라우드_사업.hwp", "page_number": 2},
      {"merged_page_content": "마이그레이션 계획", "filename": "클라우드_사업.hwp", "page_number": 7},
    ],
    [ # File 3 (missing filename)
      {"merged_page_content": "This chunk has no filename.", "filename": "", "page_number": 1},
    ],
    [ # File 4 (empty content)
      {"merged_page_content": "", "filename": "empty_content.pdf", "page_number": 1},
    ]
  ]

  # Create the database and retriever
  document_db, bm25_retriever = create_document_db(dummy_pre_chunked_data, dummy_metadata_df, dummy_embeddings, test_db_dir)

  if document_db and bm25_retriever:
    print("\nDocument database and BM25 Retriever created successfully.")
    print(f"Number of documents in Chroma DB: {document_db._collection.count()}")
    print("\nTesting BM25 Retriever...")
    test_query = "시스템 구축 요구사항"
    retrieved_docs = bm25_retriever.get_relevant_documents(test_query)
    print(f"BM25 retrieved {len(retrieved_docs)} documents for query '{test_query}'")
  
  if os.path.exists(test_db_dir):
    print(f"Attempting to clean up test directory: {test_db_dir}")
    try:
      import shutil
      shutil.rmtree(test_db_dir)
      print(f"Cleaned up {test_db_dir}")
    except PermissionError as e:
      print(f"❗ PermissionError during cleanup: {e}")
      print("Please manually delete the test directory if it persists.")
    except Exception as e:
      print(f"❗ Error during cleanup: {e}")