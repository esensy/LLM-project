import torch
import argparse
from config.settings import MUP_FILE_PATH, PLUMBER_FILE_PATH, METADATA_PATH, LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_SEED, MAX_CONVERSATION_HISTORY, DEFAULT_TOP_K
from models.embeddings import load_embedding_model
from data import load_metadata, load_documents
from database import create_metadata_db, create_document_db
from conversation import ConversationManager
from search import optimized_process_query
from utils import print_divider
from langchain_openai import ChatOpenAI

def _display_results(query, result):
    """결과 출력 함수"""
    # 재구성된 질문 출력 (변경된 경우에만)
    if "reformulated_query" in result and result["reformulated_query"] != query:
        print(f"\n\033[96m=== 재구성된 질문 ===\033[0m")
        print(f"원본: {query}")
        print(f"재구성: {result['reformulated_query']}")

    # 처리 방법 출력
    print(f"\n\033[95m=== 처리 방법 ===\033[0m")
    print(result.get("processing_method", "알 수 없음"))

    # 사용된 DB 정보 출력
    print(f"\n\033[96m=== 사용된 DB ===\033[0m")
    print(result.get("db_used", "Unknown"))

    # 응답 출력
    print("\n\033[91m=== 응답 ===\033[0m")
    print(result["response"])

    # 관련 파일 정보 출력
    print("\n\033[93m=== 관련 파일 ===\033[0m")
    for file in result["relevant_files"]:
        print(f"- {file}")

    # 메타데이터 컨텍스트 출력
    print("\n\033[95m=== 메타데이터 컨텍스트 ===\033[0m")
    if "metadata_context" in result:
        metadata_context = result["metadata_context"]
        print(metadata_context)
    else:
        print("메타데이터 컨텍스트가 없습니다.")

    # 참조 청크 출력
    print("\n\033[94m=== 참조한 청크 정보 ===\033[0m")
    if result["referenced_chunks"]:
        for i, chunk in enumerate(result["referenced_chunks"], 1):
            print(f"\n\033[92m--- 참조 청크 {i} ---\033[0m")
            print(f"사업명: {chunk['사업명']}")
            print(f"발주기관: {chunk['발주기관']}")
            print(f"유사도 점수: {chunk['similarity_score']:.4f}")

            # 청크 내용 요약 출력 (너무 길면 잘라서 출력)
            content = chunk["content"]
            if len(content) > 500:
                print(f"내용 미리보기: {content[:500]}...")
                print(f"(총 {len(content)}자)")
            else:
                print(f"내용: {content}")
    else:
        print("참조한 청크가 없습니다.")

    # 이전 참조 청크 출력 (Primary DB에서 검색한 결과)
    if "primary_referenced_chunks" in result and result["primary_referenced_chunks"]:
        print("\n\033[94m=== 이전 참조 청크 정보 (Primary DB) ===\033[0m")
        for i, chunk in enumerate(result["primary_referenced_chunks"], 1):
            print(f"\n\033[92m--- 이전 참조 청크 {i} ---\033[0m")
            print(f"사업명: {chunk['사업명']}")
            print(f"발주기관: {chunk['발주기관']}")
            print(f"유사도 점수: {chunk['similarity_score']:.4f}")

            # 청크 내용 요약 출력 (너무 길면 잘라서 출력)
            content = chunk["content"]
            if len(content) > 500:
                print(f"내용 미리보기: {content[:500]}...")
                print(f"(총 {len(content)}자)")
            else:
                print(f"내용: {content}")

def _save_conversation_log(conversation_manager):
    """대화 로그 저장 함수"""
    with open("conversation_log.txt", "w", encoding="utf-8") as f:
        f.write("=== 질문-응답 로그 ===\n\n")
        for i, log in enumerate(conversation_manager.qa_logs, 1):
            f.write(f"[대화 {i}]\n")
            f.write(f"질문: {log['query']}\n")

            # 안전한 재구성된 질문 처리
            reformulated = log.get('reformulated_query')
            if reformulated and reformulated != log['query']:
                f.write(f"재구성된 질문: {reformulated}\n")

            f.write(f"응답: {log['response']}\n\n")
            f.write("-" * 80 + "\n\n")

def optimized_main(mode="predefined", custom_questions=None):
    """
    LLM 호출을 최적화한 메인 함수
    
    Args:
        mode (str): 실행 모드 - "predefined" (사전정의 질문) 또는 "interactive" (대화형)
        custom_questions (list, optional): 사용자 정의 질문 리스트
    """
    # 1. 임베딩 모델 로드
    print("임베딩 모델 로드 중...")
    embeddings = load_embedding_model()

    # 2. 메타데이터 로드
    print("메타데이터 로드 중...")
    metadata_df = load_metadata(METADATA_PATH)

    # 3. 문서 로드
    print("문서 로드 중...")
    mup_chunks = load_documents(MUP_FILE_PATH)
    plumber_chunks = load_documents(PLUMBER_FILE_PATH)

    # 4. 메타데이터 벡터 DB 생성
    print("메타데이터 벡터 DB 생성 중...")
    metadata_db = create_metadata_db(metadata_df, embeddings)

    # 5. 문서 벡터 DB 생성
    print("문서 벡터 DB 생성 중...")
    mup_document_db = create_document_db(mup_chunks, metadata_df, embeddings, "mup_document_db")
    plumber_document_db = create_document_db(plumber_chunks, metadata_df, embeddings, "plumber_document_db")

    # 6. 대화 맥락 관리자 초기화 (메타데이터 전달)
    conversation_manager = ConversationManager(max_history=MAX_CONVERSATION_HISTORY, metadata_df=metadata_df)

    # 7. ChatOpenAI 객체 생성
    chat = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED  # 고정된 시드 값 추가
    )

    # 8. 실행 모드에 따른 질의응답 처리
    print("\n=== 공공입찰 공고 문서 질의응답 시스템 시작 ===")
    
    if mode == "predefined":
        print("=== 사전 정의된 질문으로 실행 중 ===")
        
        # 사용자 정의 질문이 있으면 사용, 없으면 기본 질문 사용
        if custom_questions:
            questions = custom_questions
        else:
            questions = [
                "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 알려줘.",
                "콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘.",
                "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?",
                "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?",
                "그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘.",
                "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.",
                "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
                "고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나? 문서를 기반으로 정확하게 답변해 줘",
                "공고 번호가 20240821893인 사업 이름이 뭐야?",
                "전체 사업 중 규모가 제일 큰 건 뭐야?", 
                "국민연금공단에서 하는 모든 사업을 알려줘.", 
                "가장 최근에 공개된 사업을 알려줘.",
                "입찰 시작일이 2025년 이후의 사업을 알려줘.",
                "입찰 마감일이 2024년 12월인 사업을 알려줘.",
                "사업 금액이 가장 큰 사업은 뭐야?",
                "가장 많은 자금이 배정된 항목은 뭔가요?",
                "예산 규모가 가장 높은 사업은 어떤 거야?",
                "가장 고비용으로 진행된 사업은 뭔데?",
                "대학 산학 협력 활동 실태조사 사업에서 요구사항 고유번호가 TER-001의 세부내용을 알려줘.",
                "대학 산학 협력 활동 실태조사 사업에서 요구사항 명칭이 SW 사업정보 제출의 고유번호를 알려줘.",
                "한국한의학연구원 통합정보 시스템 고도화 용역 사업의 모든 요구사항번호와 요구사항명을 알려줘.",
                "한국한의학연구원 통합정보 시스템 고도화 용역 사업의 모든 요구 사항 목록을 알려줘."
            ]

        for query in questions:
            print(f"질문 : {query}")
            print_divider("새 질문")
            
            # 질문 처리 및 응답 생성 (최적화된 함수 사용)
            result = optimized_process_query(
                query,
                metadata_db,
                mup_document_db,
                plumber_document_db,
                embeddings,
                metadata_df,
                conversation_manager,
                chat,
                top_k=DEFAULT_TOP_K
            )
            
            _display_results(query, result)
    
    elif mode == "interactive":
        print("=== 대화형 모드로 실행 중 ===")
        print("질문을 입력하세요. 종료하려면 'exit'를 입력하세요.")
        
        while True:
            print_divider("새 질문")
            query = input("\n질문: ")
            if query.lower() == 'exit':
                break

            # 질문 처리 및 응답 생성 (최적화된 함수 사용)
            result = optimized_process_query(
                query,
                metadata_db,
                mup_document_db,
                plumber_document_db,
                embeddings,
                metadata_df,
                conversation_manager,
                chat,
                top_k=DEFAULT_TOP_K
            )
            
            _display_results(query, result)

    # 대화 로그를 txt 파일로 저장
    _save_conversation_log(conversation_manager)

def main():
    """명령행 인수를 처리하는 메인 함수"""
    parser = argparse.ArgumentParser(description='공공입찰 공고 문서 질의응답 시스템')
    parser.add_argument('--mode', choices=['predefined', 'interactive'], default='predefined',
                       help='실행 모드: predefined (사전정의 질문), interactive (대화형)')
    parser.add_argument('--questions', nargs='*', 
                       help='사용자 정의 질문 리스트 (predefined 모드에서만 사용)')
    
    args = parser.parse_args()
    
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    print("=== LLM 호출 최적화된 공공입찰 공고 문서 질의응답 시스템 ===")
    
    # 사용자가 지정한 모드와 질문으로 실행
    optimized_main(mode=args.mode, custom_questions=args.questions)

# 스크립트 실행
if __name__ == "__main__":
    main()