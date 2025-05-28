import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings

# 임베딩 모델 로드 (한국어 특화 모델).
def load_embedding_model(model_name: str) -> Embeddings:
  """한국어 특화 임베딩 모델 로드"""
  # 서울대학교 NLP 연구실의 한국어 특화 SBERT 모델 사용.
  model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
  encode_kwargs = {"normalize_embeddings": True}

  embeddings = HuggingFaceEmbeddings(
      model_name = model_name,
      model_kwargs = model_kwargs,
      encode_kwargs = encode_kwargs
  )

  return embeddings

if __name__ == "__main__":
  print("Loading embedding model for testing....")
  embedding_model = load_embedding_model()
  print("Embedding model loaded successfully.")