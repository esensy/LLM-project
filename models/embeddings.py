import torch
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_NAME_FALLBACK, DEVICE
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

def load_embedding_model():
    """한국어 특화 임베딩 모델 로드 (GPU 메모리 최적화)"""
    
    # GPU 메모리 확인 및 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # GPU 메모리 상태 체크
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        
        print(f"GPU 총 메모리: {total_memory:.1f}GB")
        print(f"현재 사용 중인 메모리: {allocated_memory:.1f}GB")
        
        # 메모리가 부족하면 더 가벼운 모델 사용
        if total_memory < 4.0:  # 4GB 미만이면 가벼운 모델 사용
            model_name = EMBEDDING_MODEL_NAME_FALLBACK
            print(f"GPU 메모리 부족으로 가벼운 모델 사용: {model_name}")
        else:
            model_name = EMBEDDING_MODEL_NAME
            print(f"충분한 GPU 메모리, 주 모델 사용: {model_name}")
    else:
        model_name = EMBEDDING_MODEL_NAME_FALLBACK
        print("CUDA를 사용할 수 없어 CPU용 가벼운 모델을 사용합니다.")
    
    # 첫 번째 시도: 주 모델 또는 선택된 모델
    try:
        print(f"모델 로드 시도: {model_name}")
        
        # 기본 모델 설정
        model_kwargs = {
            'device': DEVICE,
            'trust_remote_code': True
        }
        
        # 안전한 인코딩 설정 (문제가 되는 파라미터 제거)
        encode_kwargs = {
            'normalize_embeddings': True
        }
        
        # 배치 크기 설정 (GPU 메모리에 따라 조절)
        if DEVICE == 'cuda' and torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_memory < 4.0:
                encode_kwargs['batch_size'] = 8  # 작은 GPU 메모리용
            else:
                encode_kwargs['batch_size'] = 16
        else:
            encode_kwargs['batch_size'] = 32  # CPU용
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        print(f"임베딩 모델 로드 완료: {model_name}")
        print(f"사용 디바이스: {DEVICE}")
        print(f"배치 크기: {encode_kwargs['batch_size']}")
        
        # GPU 메모리 상태 재확인
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            print(f"모델 로드 후 GPU 메모리: {allocated_after:.1f}GB")
        
        return embeddings
        
    except Exception as e:
        print(f"주 모델 로드 실패: {e}")
        
        # 두 번째 시도: 대안 모델 (CPU)
        if model_name != EMBEDDING_MODEL_NAME_FALLBACK:
            print("대안 모델로 재시도...")
            
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            try:
                fallback_model_kwargs = {
                    'device': 'cpu',  # CPU로 강제 변경
                    'trust_remote_code': True
                }
                
                fallback_encode_kwargs = {
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME_FALLBACK,
                    model_kwargs=fallback_model_kwargs,
                    encode_kwargs=fallback_encode_kwargs
                )
                
                print(f"대안 모델 로드 완료 (CPU): {EMBEDDING_MODEL_NAME_FALLBACK}")
                return embeddings
                
            except Exception as e2:
                print(f"대안 모델도 실패: {e2}")
        
        # 세 번째 시도: 최소 설정
        print("최소 설정으로 재시도...")
        
        try:
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 가장 단순한 설정
            simple_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}  # CPU 강제 사용
            )
            
            print("최소 설정 모델 로드 완료 (CPU)")
            return simple_embeddings
            
        except Exception as e3:
            print(f"최소 설정도 실패: {e3}")
            raise RuntimeError(f"모든 임베딩 모델 로드 시도 실패. 마지막 에러: {e3}")

def get_model_info():
    """현재 로드된 모델 정보 반환"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": DEVICE,
        "primary_model": EMBEDDING_MODEL_NAME,
        "fallback_model": EMBEDDING_MODEL_NAME_FALLBACK
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "allocated_memory_gb": torch.cuda.memory_allocated(0) / 1024**3
        })
    
    return info