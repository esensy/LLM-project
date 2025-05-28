# retrieval_utils.py

import torch
from typing import List, Dict, Any, Tuple

from langchain_openai import ChatOpenAI # For llm type hinting or if used directly
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma # For document_db type hinting
from langchain.embeddings.base import Embeddings # For embeddings type hinting
from langchain.retrievers import BM25Retriever, EnsembleRetriever # For retriever type hinting
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import CrossEncoder # If re_rank_results uses this directly

from 제안요청서.config import RERANKER_MODEL_NAME

def re_rank_results(query: str, docs_with_scores: List[Tuple[Document, float]], top_k: int, model_name: str = RERANKER_MODEL_NAME) -> List[Tuple[Document, float]]:
  # CrossEncoder 초기화.
  reranker = CrossEncoder(model_name, device = "cuda" if torch.cuda.is_available() else "cpu")

  # 문서와 점수 분리.
  docs, initial_scores = zip(*docs_with_scores)

  # Re-ranker 점수 계산.
  pairs = [(query, doc.page_content) for doc in docs]
  reranker_scores = reranker.predict(pairs)

  # 점수 합치기.
  combined_scores = [0.3 * initial + 0.7 * reranker for initial, reranker in zip(initial_scores, reranker_scores)]

  # combined_scores 기준 역순 정렬.
  scored_docs = sorted(zip(docs, combined_scores), key = lambda x: x[1], reverse = True)

  return scored_docs[:top_k]

def hybrid_retrieval(llm, query: str, document_db, bm25_retriever, top_k: int, filter_dict: dict = None) -> List[Tuple[Document, float]]:
  bm25_retriever.k = top_k # 상위 top_k개의 BM25 결과.

  # Retriever 초기화.
  chroma_retriever = document_db.as_retriever(search_kwargs = {"k": top_k * 2, "filter": filter_dict} if filter_dict else {"k": top_k * 2})
  chroma_docs_with_scores = document_db.similarity_search_with_score(query, k = top_k * 2, filter = filter_dict)

  multi_retriever = MultiQueryRetriever.from_llm(retriever = chroma_retriever, llm = llm)

  # EnsembleRetriever 사용.
  ensemble = EnsembleRetriever(
      retrievers = [bm25_retriever, multi_retriever],
      weights = [0.6, 0.4]  # 각 retriever의 가중치: bm25_retriever는 60%, 기존 chroma_retriever는 40%.
  )

  # EnsembleRetriever 결과.
  docs = ensemble.get_relevant_documents(query)

  # 점수 합치기.
  scored_results = []
  chroma_score_map = {doc.page_content: score for doc, score in chroma_docs_with_scores}

  for doc in docs:
    if doc.page_content in chroma_score_map:
      scored_results.append((doc, chroma_score_map[doc.page_content]))
    else:  # BM25Retriever는 기본 점수 부여.
      scored_results.append((doc, 1.0))

  return scored_results[:top_k]

def hybrid_multi_query_retrieval_with_reranker(llm: ChatOpenAI, query: str, document_db, bm25_retriever, top_k: int, filter_dict: dict = None, rerank: bool = True) -> List[Tuple[Document, float]]:
  docs_with_scores = hybrid_retrieval(llm, query, document_db, bm25_retriever, top_k, filter_dict)
  if not rerank:
    return docs_with_scores[:top_k]  # (docs, scores).

  reranked = re_rank_results(query, docs_with_scores, top_k)
  return reranked  # (docs, scores).

def history_aware_wrapper(llm: ChatOpenAI, query: str, document_db, bm25_retriever, top_k: int, filter_dict: dict = None, rerank: bool = True, chat_history: list = None) -> List[Tuple[Document, float]]:
  lc_chat_history = []
  if chat_history:
    for turn in chat_history:
      if "human" in turn:
        lc_chat_history.append(HumanMessage(content = turn["human"]))
      if "ai" in turn:
        lc_chat_history.append(AIMessage(content = turn["ai"]))

  # 프롬프트 템플릿 초기화.
  prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
      ("human", "대화 기록을 바탕으로 검색 쿼리를 개선하세요:")
  ])

  # Retrieval 함수.
  def base_retriever(q: str) -> List[Document]:
    scored_results = hybrid_multi_query_retrieval_with_reranker(
        llm = llm,
        query = q,
        document_db = document_db,
        bm25_retriever = bm25_retriever,
        top_k = top_k,
        filter_dict = filter_dict,
        rerank = rerank
    )
    for doc, score in scored_results:
      doc.metadata["retrieval_score"] = score
    return [doc for doc, _ in scored_results]

  history_aware_retriever = create_history_aware_retriever(llm = llm, prompt = prompt, retriever = RunnableLambda(base_retriever))
  docs = history_aware_retriever.invoke({"input": query, "chat_history": lc_chat_history })

  return [(doc, doc.metadata.get("retrieval_score", 0.0)) for doc in docs]