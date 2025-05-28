# query.py

import re
import pandas as pd
from typing import List, Dict, Any, Tuple

from langchain_openai import ChatOpenAI 
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document 
from langchain.vectorstores import Chroma 
from langchain.embeddings.base import Embeddings 

from 제안요청서.utils.utils import clean_filename, extract_numeric_amount, preprocess_metadata
from 제안요청서.model.retrieval_utils import history_aware_wrapper

def format_metadata_results(filtered_data, filter_description):
  """필터링된 메타데이터 결과를 포맷팅"""
  context = f"\n=== 메타데이터 검색 결과 ({filter_description}) ===\n"

  if len(filtered_data) > 10:
    context += f"\n총 {len(filtered_data)}개 결과 중 10개 표시\n"

  # 주요 필드 정의.
  priority_fields = ["사업명", "발주 기관", "사업 요약", "사업 금액", "공고 번호",
                     "공고 차수", "공개 일자", "입찰 참여 시작일", "입찰 참여 마감일", "파일명"]

  # 각 결과에 대한 정보 추가.
  for idx, row in filtered_data.iterrows():
    context += f"\n## 결과 {idx+1} ##\n"

    # 주요 필드 우선 출력.
    for field in priority_fields:
      if field in row and not pd.isna(row[field]) and row[field]:
        context += f"{field}: {row[field]}\n"

  return context

def process_metadata_query(query, df):
  """규칙 기반 메타데이터 질문 처리"""
  
  # 공고 번호 검색 (최우선).
  번호_match = re.search(r"공고\s*번호\s*(?:가|이)?\s*(\d{5,})", query) or re.search(r"(\d{8,})", query)
  if 번호_match:
    번호 = 번호_match.group(1)
    filtered = df[df["공고 번호"].astype(str).str.contains(번호, na = False)]
    return filtered, f"공고 번호: {번호}."

  # 사업 금액 관련 조건.
  if "사업 금액" in query or "예산" in query or "비용" in query:
    # 최대/최소 금액 질문.
    if any(keyword in query for keyword in ["가장 큰", "가장 많은", "최대", "최고"]):
      result = df.sort_values(by = "사업 금액_숫자", ascending = False).head(3)
      return result, "사업 금액 내림차순."

    elif any(keyword in query for keyword in ["가장 작은", "가장 적은", "최소", "최저"]):
      # 0원은 제외 (금액 정보가 없는 경우).
      non_zero = df[df["사업 금액_숫자"] > 0]
      result = non_zero.sort_values(by = "사업 금액_숫자", ascending = True).head(3)
      return result, "사업 금액 오름차순."

    # 금액 범위 질문.
    금액_match = re.search(r"(\d[\d,]*)(?:\s*만|\s*천만|\s*억)?(?:\s*원)?(?:\s*이상|\s*이하|\s*초과|\s*미만)", query)
    if 금액_match:
      full_match = 금액_match.group(0)
      raw_amount = 금액_match.group(1).replace(",", "")
      amount = int(raw_amount)

      # 단위 변환.
      if "억" in full_match:
        amount *= 100000000
      elif "천만" in full_match:
        amount *= 10000000
      elif "만" in full_match:
        amount *= 10000

      # 비교 조건
      if "이상" in full_match:
        result = df[df["사업 금액_숫자"] >= amount]
        return result, f"사업 금액 {amount}원 이상."
      elif "초과" in full_match:
        result = df[df['사업 금액_숫자'] > amount]
        return result, f"사업 금액 {amount}원 초과."
      elif "이하" in full_match:
        result = df[df["사업 금액_숫자"] <= amount]
        return result, f"사업 금액 {amount}원 이하."
      elif "미만" in full_match:
        result = df[df["사업 금액_숫자"] < amount]
        return result, f"사업 금액 {amount}원 미만."

  # 발주 기관 검색.
  기관_keywords = ["기관", "발주", "에서", "기관이", "기관에서"]
  if any(keyword in query for keyword in 기관_keywords):
    # 전체 기관 목록 요청.
    if "모든" in query or "전체" in query or "리스트" in query or "목록" in query:
      result = df.drop_duplicates(subset = ["발주 기관"]).sort_values(by = "발주 기관")
      return result, "모든 발주 기관 목록."

    # 특정 기관 검색.
    for org in df["발주 기관"].dropna().unique():
      if org in query:
        result = df[df["발주 기관"] == org]
        return result, f"발주 기관: {org}."

  # 날짜 관련 질문 (공개 일자).
  if "공개" in query or "공개 일자" in query:
    if "최근" in query or "가장 최근" in query:
      result = df.sort_values(by = "공개 일자_datetime", ascending = False).head(3)
      return result, "최근 공개 일자 순"
    elif "먼저" in query or "처음" in query or "가장 먼저" in query:
      result = df.sort_values(by = "공개 일자_datetime", ascending = True).head(3)
      return result, "오래된 공개 일자 순."

  # 입찰 일자 관련 질문.
  if "입찰" in query:
    # 입찰 시작일 질문.
    if "시작일" in query or "시작하" in query:
      년도_match = re.search(r"(\d{4})년", query)
      if 년도_match:
        year = 년도_match.group(1)
        result = df[df["입찰 참여 시작일_datetime"] >= f"{year}-01-01"]
        return result, f"{year}년 이후 입찰 시작 사업."

    # 입찰 마감일 질문.
    if "마감일" in query or "마감되는" in query:
      년월_match = re.search(r"(\d{4})년\s*(\d{1,2})월", query)
      if 년월_match:
        year, month = 년월_match.groups()
        month_int = int(month)

        # 월이 유효한지 확인.
        if 1 <= month_int <= 12:
          month_padded = str(month_int).zfill(2)
          next_month_int = (month_int % 12) + 1
          next_year = int(year) + (1 if month_int == 12 else 0)
          next_month_padded = str(next_month_int).zfill(2)

          start_date = f"{year}-{month_padded}-01"
          end_date = f"{next_year}-{next_month_padded}-01"

          mask = (df["입찰 참여 마감일_datetime"] >= start_date) & (df["입찰 참여 마감일_datetime"] < end_date)
          result = df[mask]
          return result, f"{year}년 {month}월 입찰 마감 사업."

  # 처리할 수 없는 메타데이터 질문.
  return None

def is_agency_content_query(query):
  """
  발주 기관과 내용 관련된 질문인지 확인
  예: "국민연금공단이 발주한, 이러닝시스템 관련 사업 요구사항을 정리해 줘."
  """
  # 발주 기관 관련 키워드.
  agency_keywords = ["발주한", "에서 하는", "에서 진행", "에서 발주", "기관의", "기관에서"]

  # 내용 관련 키워드.
  content_keywords = ["요구사항", "시스템", "기능", "서비스", "관련 사업", "내용", "정리", "알려줘",
                      "분석", "요약", "관련된", "필요한", "필수", "상세", "자세히"]

  # 발주 기관 키워드와 내용 키워드가 모두 포함된 경우.
  has_agency = any(keyword in query for keyword in agency_keywords)
  has_content = any(keyword in query for keyword in content_keywords)

  return has_agency and has_content

def generate_response(chat, query, context_text, chat_history: list = None):
  """컨텍스트를 바탕으로 응답 생성"""
  # 간단한 템플릿 사용.

  system_template = """
  당신은 공공 입찰 공고 정보를 제공하는 시스템이며 대한민국 공공기관의 제안요청서를 분석하는 전문 AI 비서입니다.
  각 문서는 해당 메타데이터(공고 번호, 사업명, 사업 금액 등)를 포함합니다. 이 정보를 적극적으로 반영해 질문에 답변하길 바랍니다.
  사용자 질문에 대해 제공된 메타데이터 정보를 바탕으로 간결하고 정확한 답변을 제공하길 바랍니다.
  메타데이터에 포함된 내용만을 바탕으로 답변하며, 검색된 메타데이터가 없으면 그렇게 알려주길 바랍니다.
  사업 금액과 관련된 질문은 VAT를 포함하여 대답하고, 답변에 VAT를 포함했다는 정보를 함께 제공하길 바랍니다.
  사업 요구사항과 관련된 질문에 대해서는, 문서에 제공된 요구사항 상세 부분을 참고하여 요구사항 고유번호와 함께(고유번호가 존재하다면) 답변하길 바랍니다.
  대화 기록과 제공된 컨텍스트(맥락)을 활용하길 바랍니다:
  {history_context}

  제공된 메타데이터:
  {context}
  """

  history_context = "\n".join([f"사용자: {msg['human']}\n인공지능: {msg['ai']}" for msg in chat_history]) if chat_history else "기록 없음."

  chat_prompt = ChatPromptTemplate.from_messages([
      SystemMessagePromptTemplate.from_template(system_template),
      HumanMessagePromptTemplate.from_template("{question}")
  ])

  response = chat.invoke(
      chat_prompt.format_messages(
          history_context = history_context,
          context = context_text,
          question = query
      )
  )

  return response.content

def process_agency_content_query(chat, query, metadata_df, metadata_db, document_db, bm25_retriever, embeddings, top_k = 5, chat_history: list = None):
  """
  발주 기관 + 내용 관련 하이브리드 질문 처리.
  1. 먼저 메타데이터에서 발주 기관 관련 정보로 관련 파일 선정.
  2. 선정된 파일 내에서 내용 관련 질의 수행.
  3. 결과 통합하여 반환.
  """
  # 1. 발주 기관 추출.
  agency = None

  # 발주 기관명 추출 패턴.
  agency_patterns = [
      r"([\w가-힣]+)(?:이|에서|가)\s*발주한",
      r"([\w가-힣]+)(?:에서|의|이|가)\s*하는",
      r"([\w가-힣]+)(?:에서|의|이|가)\s*진행"
  ]

  for pattern in agency_patterns:
    match = re.search(pattern, query)
    if match:
      agency = match.group(1)
      break

  # 발주 기관을 찾지 못한 경우, 기존 메타데이터에서 존재하는 기관명 탐색.
  if not agency:
    for org in metadata_df["발주 기관"].dropna().unique():
      if org in query:
        agency = org
        break

  if not agency:
    print("발주 기관을 찾을 수 없습니다. 일반 검색으로 진행합니다.")
    # 기존 일반 검색 로직으로 진행.
    return None

  print(f"발주 기관 식별됨: {agency}")

  # 2. 해당 발주 기관의 파일 목록 추출.
  agency_files = metadata_df[metadata_df["발주 기관"].str.contains(agency, na = False)]["파일명"].tolist()

  if not agency_files:
    print(f"{agency} 관련 파일을 찾을 수 없습니다.")
    # 기존 일반 검색 로직으로 진행.
    return None

  print(f"발주 기관 관련 파일 {len(agency_files)}개 식별됨.")

  # 3. 해당 파일 내에서 내용 검색.
  # 먼저 메타데이터 검색으로 관련 파일 점수 부여.
  metadata_results = metadata_db.similarity_search_with_score(query, k = 3)

  # 메타데이터 검색 결과에서 파일명만 추출.
  metadata_files = []
  for doc, score in metadata_results:
    filename = doc.metadata.get("파일명", "")
    if filename:
      metadata_files.append(filename)

  print(f"메타데이터 검색 관련 파일: {metadata_files}.")

  # 발주 기관 파일과 메타데이터 검색 결과의 교집합 우선.
  priority_files = list(set(agency_files).intersection(set(metadata_files)))

  # 교집합이 없으면 발주 기관 파일 전체 사용.
  if not priority_files:
    priority_files = agency_files

  print(f"우선 검색 파일: {priority_files}")

  # 4. 선정된 파일 내에서 관련 청크 검색.
  filter_dict = {"filename": {"$in": priority_files}}

  # 해당 파일 내에서 내용 검색.
  # document_results = document_db.similarity_search_with_score(query, k = top_k, filter = filter_dict)
  document_results = history_aware_wrapper(chat, query, document_db, bm25_retriever, top_k, filter_dict = filter_dict, rerank = True, chat_history = chat_history or [])

  # 5. 검색 결과 정보 추출.
  contexts = []
  for i, (doc, score) in enumerate(document_results):
    contexts.append({
        "content": doc.page_content,
        "사업명": doc.metadata.get("사업명", ""),
        "발주기관": doc.metadata.get("발주기관", ""),
        "filename": doc.metadata.get("filename", ""),
        "score": score,
        "project_context": "하이브리드 검색"
    })

  # 6. 선정된 파일의 모든 메타데이터 수집.
  file_metadata_dict = {}
  for file in priority_files:
    # 파일명으로 메타데이터 찾기.
    file_rows = metadata_df[metadata_df["파일명"] == file]
    if not file_rows.empty:
      row = file_rows.iloc[0]  # 첫 번째 매칭 행 사용.
      file_metadata_dict[file] = row.to_dict()

  # 7. 컨텍스트 구성.
  # 메타데이터 컨텍스트 구성.
  metadata_context = "\n=== 발주 기관 정보 ===\n"
  for file, metadata in file_metadata_dict.items():
    metadata_context += f"\n## 파일: {file} ##\n"
    priority_fields = ["사업명", "발주 기관", "사업 요약", "사업 금액", "공고 번호", "공고 차수", "공개 일자", "입찰 참여 시작일", "입찰 참여 마감일"]
    for field in priority_fields:
      if field in metadata and not pd.isna(metadata[field]) and metadata[field]:
        metadata_context += f"{field}: {metadata[field]}\n"

  # 내용 컨텍스트 구성.
  content_context = "\n\n=== 문서 내용 ===\n"
  for i, ctx in enumerate(contexts, 1):
    content_context += f"\n--- 문서 {i} ---\n"
    content_context += f"사업명: {ctx['사업명']}\n"
    content_context += f"발주기관: {ctx['발주기관']}\n"
    content_context += f"내용: {ctx['content']}\n"

  # 전체 컨텍스트.
  context_text = metadata_context + content_context

  # 8. 응답 생성.
  # 하이브리드 검색에 맞는 시스템 프롬프트 구성.
  system_template = """
  당신은 공공 입찰 공고 정보를 제공하는 시스템이며 대한민국 공공기관의 제안요청서를 분석하는 전문 AI 비서입니다.
  각 문서는 해당 메타데이터(공고 번호, 사업명, 사업 금액 등)를 포함합니다. 이 정보를 적극적으로 반영해 질문에 답변하길 바랍니다.
  발주 기관 정보를 통해 사업의 개요, 규모, 일정 등을 파악하고, 문서 내용을 통해 구체적인 요구사항, 기능, 서비스 내용을 파악하길 바랍니다.
  두 정보를 종합하여 사용자 질문에 대한 통합적인 답변을 제공하길 바랍니다.
  사용자 질문에 대해 제공된 메타데이터 정보를 바탕으로 간결하고 정확한 답변을 제공하길 바라고 제공된 발주 기관 정보와 문서 내용을 함께 활용하길 바랍니다.
  메타데이터에 포함된 내용만을 바탕으로 답변하며, 검색된 메타데이터가 없으면 그렇게 알려주길 바랍니다.
  문서 정보에 관련 내용이 없으면 "해당 정보가 제공된 문서에 없습니다."라고 답변하길 바랍니다.
  사업 금액과 관련된 질문은 VAT를 포함하여 대답하고, 답변에 VAT를 포함했다는 정보를 함께 제공하길 바랍니다.
  사업 요구사항과 관련된 질문에 대해서는, 문서에 제공된 요구사항 상세 부분을 참고하여 고유번호와 함께(고유번호가 존재하다면) 답변하길 바랍니다.
  대화 기록과 제공된 컨텍스트(맥락)을 활용하길 바랍니다:

  제공된 정보:
  {context}
  """

  human_template = "{question}"

  # 프롬프트 템플릿 생성.
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  # 응답 생성.
  response = chat.invoke(
      chat_prompt.format_messages(
          context = context_text,
          question = query
      )
  )

  response_content = response.content

  # 9. 결과 반환.
  return {
      "query": query,
      "relevant_files": priority_files,
      "contexts": contexts,
      "response": response_content,
      "chat_history": chat_history + [{"human": query, "ai": response_content}],  # 기록 업데이트.
      "file_metadata": file_metadata_dict,
      "metadata_context": metadata_context,
      "referenced_chunks": [
          {
              "content": ctx["content"],
              "사업명": ctx["사업명"],
              "발주기관": ctx["발주기관"],
              "similarity_score": ctx["score"],
              "project_context": ctx.get("project_context", "하이브리드 검색")
          } for ctx in contexts
      ]
  }

# 질문 처리 및 응답 생성 함수.
def process_query(chat, query, metadata_db, document_db, bm25_retriever, embeddings, metadata_df, top_k = 5, re_rank = True, chat_history: list = None):
  """
  사용자 질문을 처리하고 응답 생성.
  1. 규칙 기반으로 메타데이터 질문 감지 및 처리.
  2. 필요시 메타데이터 검색과 내용 검색을 함께 활용.
  3. 응답 생성.
  """

  # 전처리: 데이터프레임 복사 및 필요한 변환 수행.
  df = preprocess_metadata(metadata_df.copy())

  # 메타데이터 질문 처리 시도.
  metadata_result = process_metadata_query(query, df)

  # 발주 기관 관련 내용 + 메타데이터 하이브리드 검색 케이스 처리.
  if is_agency_content_query(query):
    print("발주 기관 + 내용 관련 하이브리드 질문 감지됨.")
    return process_agency_content_query(chat, query, metadata_df, metadata_db, document_db, bm25_retriever, embeddings, top_k, chat_history)

  # 일반 메타데이터 질문 처리 성공 시.
  if metadata_result is not None:
    filtered_data, filter_description = metadata_result

    if not filtered_data.empty:
      # 결과가 너무 많으면 제한.
      if len(filtered_data) > 10:
        filtered_data = filtered_data.head(10)

        # 컨텍스트 구성.
        context_text = format_metadata_results(filtered_data, filter_description)

        # 응답 생성.
        response = generate_response(chat, query, context_text, chat_history)

        return {
            "query": query,
            "relevant_files": filtered_data["파일명"].tolist() if "파일명" in filtered_data.columns else [],
            "contexts": [],
            "response": response,
            "file_metadata": {row.get('파일명', f'result_{i}'): row.to_dict() for i, (idx, row) in enumerate(filtered_data.iterrows())},
            "metadata_context": context_text,
            "referenced_chunks": []
        }

  # 1. 메타데이터 검색으로 관련 파일 선정.
  metadata_results = metadata_db.similarity_search_with_score(query, k = 3)

  relevant_files = []
  # 메타데이터 검색 결과 출력 (이전 코드와 동일).
  print("\n=== 메타데이터 검색 결과 ===")
  for i, (doc, score) in enumerate(metadata_results):
    filename = doc.metadata.get("파일명", "")
    project_name = doc.metadata.get("사업명", "")
    agency = doc.metadata.get("발주기관", "")

    # 유사도 점수와 함께 결과 출력.
    print(f"{i + 1}. 파일명: {filename}")
    print(f"   사업명: {project_name}")
    print(f"   발주기관: {agency}")
    print(f"   유사도 점수: {score:.6f}" + (" ✓" if score < 1.5 else " ✗"))

    if score < 1.5:  # 임계값 설정.
      if filename:
        relevant_files.append(filename)

    # 선정된 파일에 대한 모든 메타데이터 수집.
    file_metadata_dict = {}

    # metadata_df에서 관련 파일의 모든 메타데이터 가져오기.
    if metadata_df is not None and len(relevant_files) > 0:
      for file in relevant_files:
        # 파일명으로 메타데이터 찾기.
        file_rows = metadata_df[metadata_df["파일명"] == file]
        if not file_rows.empty:
          row = file_rows.iloc[0]  # 첫 번째 매칭 행 사용.

          # 모든 메타데이터 수집
          metadata_info = {}
          for col in metadata_df.columns:
            if col in row and not pd.isna(row[col]) and row[col]:
              metadata_info[col] = row[col]

          file_metadata_dict[file] = metadata_info

  # 관련 파일 목록 출력.
  print(f"\n선정된 파일 ({len(relevant_files)}개):")
  for file in relevant_files:
    print(f"- {file}")

  # 파일이 선정되지 않은 경우 전체 문서에서 검색.
  # 혹은, 선정된 파일 내에서 관련 청크 검색 (필터 사용).
  document_results = history_aware_wrapper(
      chat,
      query,
      document_db,
      bm25_retriever,
      top_k,
      filter_dict = {"filename": {"$in": relevant_files}} if relevant_files else None,
      rerank = True,
      chat_history = chat_history or []
  )

  # 청크 정보 추출 (이전 코드와 동일).
  contexts = []
  for i, (doc, score) in enumerate(document_results):
    contexts.append({
        "content": doc.page_content,
        "사업명": doc.metadata.get("사업명", ""),
        "발주기관": doc.metadata.get("발주기관", ""),
        "filename": doc.metadata.get("filename", ""),
        "score": score,
        "project_context": "단일 검색"
    })

  # 3. OpenAI GPT 모델을 이용한 응답 생성
  system_template = """
  당신은 공공 입찰 공고 정보를 제공하는 시스템이며 대한민국 공공기관의 제안요청서를 분석하는 전문 AI 비서입니다.
  각 문서는 해당 메타데이터(공고 번호, 사업명, 사업 금액 등)를 포함합니다. 이 정보를 적극적으로 반영해 질문에 답변하길 바랍니다.
  발주 기관 정보를 통해 사업의 개요, 규모, 일정 등을 파악하고, 문서 내용을 통해 구체적인 요구사항, 기능, 서비스 내용을 파악하길 바랍니다.
  두 정보를 종합하여 사용자 질문에 대한 통합적인 답변을 제공하길 바랍니다.
  사용자 질문에 대해 제공된 메타데이터 정보를 바탕으로 간결하고 정확한 답변을 제공하길 바라고 제공된 발주 기관 정보와 문서 내용을 함께 활용하길 바랍니다.
  메타데이터에 포함된 내용만을 바탕으로 답변하며, 검색된 메타데이터가 없으면 그렇게 알려주길 바랍니다.
  문서 정보에 관련 내용이 없으면 "해당 정보가 제공된 문서에 없습니다."라고 답변하길 바랍니다.
  사업 금액과 관련된 질문은 VAT를 포함하여 대답하고, 답변에 VAT를 포함했다는 정보를 함께 제공하길 바랍니다.
  사업 요구사항과 관련된 질문에 대해서는, 문서에 제공된 요구사항 상세 부분을 참고하여 고유번호와 함께(고유번호가 존재하다면) 답변하길 바랍니다.
  대화 기록과 제공된 컨텍스트(맥락)을 활용하길 바랍니다:

  제공된 문서 정보 및 메타데이터:
  {context}
  """

  human_template = "{question}"

  # 프롬프트 템플릿 생성.
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  # 컨텍스트 정보 문자열로 변환 (메타데이터 포함) 부분 수정.
  context_text = ""

  # 먼저 선정된 파일의 모든 메타데이터 추가.
  if file_metadata_dict:
    context_text += "\n=== 선정된 파일의 메타데이터 ===\n"
    for file, metadata in file_metadata_dict.items():
      context_text += f"\n## 파일: {file} ##\n"

      # 주요 메타데이터 항목 먼저 정렬하여 출력.
      priority_fields = ["사업명", "발주 기관", "사업 요약", "사업 금액", "공고 번호", "공고 차수", "공개 일자", "입찰 참여 시작일", "입찰 참여 마감일"]

      # 주요 항목 출력.
      for field in priority_fields:
        if field in metadata and metadata[field]:
          context_text += f"{field}: {metadata[field]}\n"

      # 나머지 항목 출력.
      for field, value in metadata.items():
        if field not in priority_fields and field != "파일명" and field != "텍스트":
          context_text += f"{field}: {value}\n"

  # 청크 내용 추가.
  context_text += "\n\n=== 문서 내용 ===\n"
  for i, ctx in enumerate(contexts, 1):
    context_text += f"\n--- 문서 {i} ---\n"
    context_text += f"사업명: {ctx['사업명']}\n"
    context_text += f"발주기관: {ctx['발주기관']}\n"
    context_text += f"내용: {ctx['content']}\n"

  metadata_preview = context_text.split("=== 문서 내용 ===")[0]  # 메타데이터 부분만 추출.

  print("\n\033[95m=== 총 컨텍스트 길이 ===\033[0m")
  print(f"메타데이터 부분: {len(metadata_preview)} 자")
  print(f"전체 컨텍스트: {len(context_text)} 자")

  # 컨텍스트가 비어있는지 확인.
  if not contexts and not file_metadata_dict:
    response_content = "해당 정보가 제공된 문서에 없습니다."
  else:
    # 응답 생성.
    response = chat.invoke(
        chat_prompt.format_messages(
            context = context_text,
            question = query
        )
    )
    response_content = response.content

  return {
      "query": query,
      "relevant_files": relevant_files,
      "contexts": contexts,
      "response": response_content,
      "chat_history": chat_history + [{"human": query, "ai": response_content}],  # 기록 업데이트.
      "file_metadata": file_metadata_dict,                               # 선정된 파일의 메타데이터 추가.
      "metadata_context": metadata_preview,                              # 메타데이터 컨텍스트 부분 추가.
      "referenced_chunks": [
          {
              "content": ctx["content"],
              "사업명": ctx["사업명"],
              "발주기관": ctx["발주기관"],
              "similarity_score": ctx["score"],
              "project_context": ctx.get("project_context", "단일 검색")
          } for ctx in contexts
      ]
  }
