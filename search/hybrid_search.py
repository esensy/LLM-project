import re
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from config.settings import LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_SEED

def is_agency_content_query(query):
    """
    발주 기관과 내용 관련된 질문인지 확인
    예: "국민연금공단이 발주한, 이러닝시스템 관련 사업 요구사항을 정리해 줘."
    """
    # 발주 기관 관련 키워드
    agency_keywords = ['발주한', '에서 하는', '에서 진행', '에서 발주', '기관의', '기관에서']

    # 내용 관련 키워드
    content_keywords = ['시스템', '기능', '서비스', '관련 사업', '내용', '정리', '알려줘',
                        '분석', '요약', '관련된', '필요한', '필수', '상세', '자세히']

    # 발주 기관 키워드와 내용 키워드가 모두 포함된 경우
    has_agency = any(keyword in query for keyword in agency_keywords)
    has_content = any(keyword in query for keyword in content_keywords)

    return has_agency and has_content

def process_agency_content_query(query, metadata_df, metadata_db, document_db, embeddings, top_k=5):
    """
    발주 기관 + 내용 관련 하이브리드 질문 처리
    1. 먼저 메타데이터에서 발주 기관 관련 정보로 관련 파일 선정
    2. 선정된 파일 내에서 내용 관련 질의 수행
    3. 결과 통합하여 반환
    """
    # 1. 발주 기관 추출
    agency = None

    # 발주 기관명 추출 패턴
    agency_patterns = [
        r'([\w가-힣]+)(?:이|에서|가)\s*발주한',
        r'([\w가-힣]+)(?:에서|의|이|가)\s*하는',
        r'([\w가-힣]+)(?:에서|의|이|가)\s*진행'
    ]

    for pattern in agency_patterns:
        match = re.search(pattern, query)
        if match:
            agency = match.group(1)
            break

    # 발주 기관을 찾지 못한 경우, 기존 메타데이터에서 존재하는 기관명 탐색
    if not agency:
        for org in metadata_df['발주 기관'].dropna().unique():
            if org in query:
                agency = org
                break

    if not agency:
        print("발주 기관을 찾을 수 없습니다. 일반 검색으로 진행합니다.")
        # 기존 일반 검색 로직으로 진행
        return None

    print(f"발주 기관 식별됨: {agency}")

    # 2. 해당 발주 기관의 파일 목록 추출
    agency_files = metadata_df[metadata_df['발주 기관'].str.contains(agency, na=False)]['파일명'].tolist()

    if not agency_files:
        print(f"{agency} 관련 파일을 찾을 수 없습니다.")
        # 기존 일반 검색 로직으로 진행
        return None

    print(f"발주 기관 관련 파일 {len(agency_files)}개 식별됨")

    # 3. 해당 파일 내에서 내용 검색
    # 먼저 메타데이터 검색으로 관련 파일 점수 부여
    metadata_results = metadata_db.similarity_search_with_score(agency, k=3)

    # 메타데이터 검색 결과에서 파일명만 추출
    metadata_files = []
    for doc, score in metadata_results:
        filename = doc.metadata.get("파일명", "")
        if filename:
            metadata_files.append(filename)

    print(f"메타데이터 검색 관련 파일: {metadata_files}")

    # 발주 기관 파일과 메타데이터 검색 결과의 교집합 우선
    priority_files = list(set(agency_files).intersection(set(metadata_files)))

    # 교집합이 없으면 발주 기관 파일 전체 사용
    if not priority_files:
        priority_files = agency_files

    print(f"우선 검색 파일: {priority_files}")

    # 4. 선정된 파일 내에서 관련 청크 검색
    filter_dict = {"filename": {"$in": priority_files}}

    # 해당 파일 내에서 내용 검색
    document_results = document_db.similarity_search_with_score(
        query, k=top_k, filter=filter_dict
    )

    # 5. 검색 결과 정보 추출
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

    # 6. 선정된 파일의 모든 메타데이터 수집
    file_metadata_dict = {}
    for file in priority_files:
        # 파일명으로 메타데이터 찾기
        file_rows = metadata_df[metadata_df['파일명'] == file]
        if not file_rows.empty:
            row = file_rows.iloc[0]  # 첫 번째 매칭 행 사용
            file_metadata_dict[file] = row.to_dict()

    # 7. 컨텍스트 구성
    # 메타데이터 컨텍스트 구성
    metadata_context = "\n=== 발주 기관 정보 ===\n"
    for file, metadata in file_metadata_dict.items():
        metadata_context += f"\n## 파일: {file} ##\n"
        priority_fields = ['사업명', '발주 기관', '사업 요약', '사업 금액',
                          '공고 번호', '공고 차수', '공개 일자',
                          '입찰 참여 시작일', '입찰 참여 마감일']

        for field in priority_fields:
            if field in metadata and not pd.isna(metadata[field]) and metadata[field]:
                metadata_context += f"{field}: {metadata[field]}\n"

    # 내용 컨텍스트 구성
    content_context = "\n\n=== 문서 내용 ===\n"
    for i, ctx in enumerate(contexts, 1):
        content_context += f"\n--- 문서 {i} ---\n"
        content_context += f"사업명: {ctx['사업명']}\n"
        content_context += f"발주기관: {ctx['발주기관']}\n"
        content_context += f"내용: {ctx['content']}\n"

    # 전체 컨텍스트
    context_text = metadata_context + content_context

    # 8. 응답 생성
    # 하이브리드 검색에 맞는 시스템 프롬프트 구성
    system_template = """
    당신은 공공입찰 공고 문서를 기반으로 정확한 정보를 제공하는 전문가입니다.
    사용자의 질문에 답하기 위해 제공된 발주 기관 정보와 문서 내용을 함께 활용하세요.

    발주 기관 정보를 통해 사업의 개요, 규모, 일정 등을 파악하고,
    문서 내용을 통해 구체적인 요구사항, 기능, 서비스 내용을 파악하세요.

    두 정보를 종합하여 사용자 질문에 대한 통합적인 답변을 제공하세요.
    문서 정보에 관련 내용이 없으면 "해당 정보가 제공된 문서에 없습니다."라고 답변하세요.

    제공된 정보:
    {context}
    """

    human_template = "{question}"

    # ChatGPT 모델 설정
    chat = ChatOpenAI(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        seed=LLM_SEED  # 고정된 시드 값 추가
    )

    # 프롬프트 템플릿 생성
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 응답 생성
    response = chat.invoke(
        chat_prompt.format_messages(
            context=context_text,
            question=query
        )
    )

    response_content = response.content

    # 9. 결과 반환
    return {
        "query": query,
        "relevant_files": priority_files,
        "contexts": contexts,
        "response": response_content,
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