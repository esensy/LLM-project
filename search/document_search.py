# 최적화된 질문 처리 함수
def optimized_process_query(query, metadata_db, primary_document_db, secondary_document_db, embeddings, metadata_df, conversation_manager, chat, top_k=5):
    """
    LLM 호출을 최적화한 질문 처리 함수

    Args:
        query (str): 사용자 질문
        metadata_db: 메타데이터 벡터 DB
        primary_document_db: 주 문서 DB (mup_document_db)
        secondary_document_db: 보조 문서 DB (plumber_document_db)
        embeddings: 임베딩 모델
        metadata_df (DataFrame): 메타데이터 데이터프레임
        conversation_manager (ConversationManager): 대화 맥락 관리자
        chat: ChatOpenAI 객체
        top_k (int): 검색할 최대 청크 수

    Returns:
        dict: 처리 결과를 담은 딕셔너리
    """
    from data.preprocessor import preprocess_metadata
    from search.metadata_search import process_metadata_query, format_metadata_results
    from search.hybrid_search import is_agency_content_query, process_agency_content_query
    from conversation.llm_processor import process_with_llm
    from response.generator import generate_response
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from config.settings import METADATA_SIMILARITY_THRESHOLD, DYNAMIC_SIMILARITY_THRESHOLD
    import pandas as pd
    import re
    import json
    
    # 전처리: 메타데이터 데이터프레임 전처리
    df = preprocess_metadata(metadata_df.copy())

    print("\n=== 질문 처리 단계 시작 ===")

    # 1. 먼저 규칙 기반 처리 시도
    print("\n=== 1. 규칙 기반 메타데이터 처리 시도 ===")
    metadata_result = process_metadata_query(query, df)

    if metadata_result is not None:
        filtered_data, filter_description = metadata_result

        if not filtered_data.empty:
            # 결과가 너무 많으면 제한
            if len(filtered_data) > 10:
                display_data = filtered_data.head(10)
            else:
                display_data = filtered_data

            # 컨텍스트 구성
            context_text = format_metadata_results(display_data, filter_description)

            # 응답 생성
            response = generate_response(query, context_text)

            print("규칙 기반 처리 성공")

            # 결과 구성
            result = {
                "query": query,
                "relevant_files": filtered_data['파일명'].tolist() if '파일명' in filtered_data.columns else [],
                "contexts": [],
                "response": response,
                "file_metadata": {row.get('파일명', f'result_{i}'): row.to_dict()
                                for i, (idx, row) in enumerate(filtered_data.iterrows())},
                "metadata_context": context_text,
                "referenced_chunks": [],
                "processing_method": "규칙 기반 메타데이터 처리"
            }

            # 대화 맥락에 추가
            conversation_manager.add_interaction(query, response, result, None)  # 또는 query
            return result

    # 2. 발주 기관 관련 내용+메타데이터 하이브리드 검색 시도
    print("\n=== 2. 발주 기관 + 내용 관련 하이브리드 처리 시도 ===")
    if is_agency_content_query(query):
        print("발주 기관 + 내용 관련 하이브리드 질문 감지됨")
        agency_result = process_agency_content_query(query, metadata_df, metadata_db, primary_document_db, embeddings, top_k)

        if agency_result:
            agency_result["processing_method"] = "하이브리드 처리"
            # 대화 맥락에 추가
            conversation_manager.add_interaction(query, agency_result["response"], agency_result, None)  # reformulated_query는 None
            return agency_result

    # 3. 대화 맥락 및 LLM 기반 처리 (LLM 호출 통합)
    print("\n=== 3. 대화 맥락 고려 및 LLM 기반 처리 ===")
    conversation_context = conversation_manager.get_conversation_context()

    # LLM 호출 - 대화 맥락 파악 및 메타데이터 처리 통합
    llm_result = process_with_llm(query, conversation_context, df, chat)

    # 분석 결과 추출
    analysis = llm_result.get("analysis", {})
    metadata_only = analysis.get("metadata_only", False)
    rule_based_query = analysis.get("rule_based_query", False)
    reformulated_query = analysis.get("reformulated_query", query)
    metadata_response = llm_result.get("metadata_response", "")

    # 분석 결과 출력
    print(f"대화 맥락 분석 결과:")
    print(f"메타데이터만으로 답변 가능: {metadata_only}")
    print(f"규칙 기반 처리 가능: {rule_based_query}")
    print(f"재구성된 질문: {reformulated_query}")
    print(f"관련 컬럼: {', '.join(analysis.get('relevant_columns', []))}")
    print(f"추출된 엔티티: {analysis.get('entities', {})}")
    print(f"분석 근거: {analysis.get('reasoning', '')}")

    # 케이스 1: 메타데이터만으로 처리 가능하고 규칙 기반 처리가 가능한 경우 (특정 사업명 불포함 & 메타데이터 질의)
    if metadata_only and rule_based_query:
        print("\n=== 3-1. 규칙 기반 메타데이터 처리 재시도 (LLM 재구성 질문 사용) ===")

        # 재구성된 질문으로 규칙 기반 처리 재시도
        meta_retry_result = process_metadata_query(reformulated_query, df)

        if meta_retry_result is not None:
            filtered_data, filter_description = meta_retry_result

            if not filtered_data.empty:
                # 결과가 너무 많으면 제한
                if len(filtered_data) > 10:
                    display_data = filtered_data.head(10)
                else:
                    display_data = filtered_data

                # 컨텍스트 구성
                context_text = format_metadata_results(display_data, filter_description)

                # 응답 생성
                response = generate_response(reformulated_query, context_text)

                print("재구성된 질문으로 규칙 기반 처리 성공")

                # 결과 구성
                result = {
                    "query": query,
                    "reformulated_query": reformulated_query,
                    "relevant_files": filtered_data['파일명'].tolist() if '파일명' in filtered_data.columns else [],
                    "contexts": [],
                    "response": response,
                    "file_metadata": {row.get('파일명', f'result_{i}'): row.to_dict()
                                    for i, (idx, row) in enumerate(filtered_data.iterrows())},
                    "metadata_context": context_text,
                    "referenced_chunks": [],
                    "processing_method": "LLM 재구성 + 규칙 기반 메타데이터 처리"
                }

                # 대화 맥락에 추가
                conversation_manager.add_interaction(query, response, result, reformulated_query)
                return result
        print("재구성된 질문으로 규칙 기반 처리 실패")

    # 케이스 2: 메타데이터만으로 처리 가능하지만 규칙 기반 처리 불가능한 경우 (특정 사업명 포함 & 일반정보 요청)
    if metadata_only and not rule_based_query and metadata_response:
        print("\n=== 3-2. LLM 기반 메타데이터 처리 (규칙 기반 처리 불가) ===")

    # 4. 일반 검색 로직 진행
    print("\n=== 4. 일반 검색 처리 ===")

    # 분석 결과에서 엔티티와 검색 키워드 추출
    entities = analysis.get("entities", {})
    search_keywords = analysis.get("search_keywords", [])
    project_agency_pairs = entities.get("project_agency_pairs", [])

    print(f"추출된 사업명-발주기관 페어: {project_agency_pairs}")
    print(f"검색 키워드: {search_keywords}")

    # 재구성된 질문 사용
    search_query = reformulated_query if reformulated_query != query else query

    # 메타데이터 검색용 쿼리 구성 및 검색 수행
    if project_agency_pairs:
        print(f"사업명-발주기관 페어 기반 검색 ({len(project_agency_pairs)}개 페어)")

        all_metadata_results = []

        # 각 사업명-발주기관 페어별로 개별 검색
        for i, pair in enumerate(project_agency_pairs):
            project_name = pair.get("사업명", "").strip()
            agency = pair.get("발주기관", "").strip()

            # 페어 쿼리 구성 (빈 값 제외)
            pair_terms = [term for term in [project_name, agency] if term]
            if pair_terms:
                pair_query = " ".join(pair_terms)
                print(f"페어 {i+1} 검색 쿼리: '{pair_query}'")

                # 개별 검색 수행
                pair_results = metadata_db.similarity_search_with_score(pair_query, k=1)
                all_metadata_results.extend(pair_results)

        # 중복 제거 및 점수 기준 정렬
        seen_files = set()
        unique_results = []
        for doc, score in sorted(all_metadata_results, key=lambda x: x[1]):
            filename = doc.metadata.get("파일명", "")
            if filename and filename not in seen_files:
                unique_results.append((doc, score))
                seen_files.add(filename)

        metadata_results = unique_results[:max(len(project_agency_pairs), 3)]  # 최대 결과 수 제한
        k_value = len(metadata_results)
        similarity_threshold = DYNAMIC_SIMILARITY_THRESHOLD

    else:
        # 엔티티가 없으면 기존 방식 사용
        metadata_search_query = search_query
        k_value = 3
        similarity_threshold = METADATA_SIMILARITY_THRESHOLD
        print(f"메타데이터 검색 쿼리 (전체 질문 기반): '{metadata_search_query}'")
        metadata_results = metadata_db.similarity_search_with_score(metadata_search_query, k=k_value)

    relevant_files = []
    # 메타데이터 검색 결과 출력
    print(f"\n메타데이터 검색 결과 (k={k_value}, 임계값={similarity_threshold}):")
    for i, (doc, score) in enumerate(metadata_results):
        filename = doc.metadata.get("파일명", "")
        project_name = doc.metadata.get("사업명", "")
        agency = doc.metadata.get("발주기관", "")

        # 유사도 점수와 함께 결과 출력
        print(f"{i+1}. 파일명: {filename}")
        print(f"   사업명: {project_name}")
        print(f"   발주기관: {agency}")
        print(f"   유사도 점수: {score:.6f}" + (" ✓" if score < similarity_threshold else " ✗"))

        if score < similarity_threshold:  # 동적 임계값 사용
            if filename:
                relevant_files.append(filename)

    # 대화 맥락에서 언급된 파일이 있으면 추가 -> 오히려 검색 품질이 저하되는 요인
    # context_files = conversation_manager.mentioned_entities.get("파일명", [])
    # for file in context_files:
    #     if file not in relevant_files:
    #         # relevant_files.append(file)
    #         print(f"xxxxx일단 안함 -> 대화 맥락에서 파일 추가: {file}")

    # 선정된 파일에 대한 모든 메타데이터 수집
    file_metadata_dict = {}

    # metadata_df에서 관련 파일의 모든 메타데이터 가져오기
    if metadata_df is not None and len(relevant_files) > 0:
        for file in relevant_files:
            # 파일명으로 메타데이터 찾기
            file_rows = metadata_df[metadata_df['파일명'] == file]
            if not file_rows.empty:
                row = file_rows.iloc[0]  # 첫 번째 매칭 행 사용
                file_metadata_dict[file] = row.to_dict()

    # 관련 파일 목록 출력
    print(f"\n선정된 파일 ({len(relevant_files)}개):")
    for file in relevant_files:
        print(f"- {file}")

    # 청크 검색 쿼리 구성 (검색 키워드 활용)
    if search_keywords:
        # 검색 키워드가 있으면 키워드 중심으로 검색
        chunk_search_query = " ".join(search_keywords)
        print(f"키워드 기반 청크 검색 쿼리: '{chunk_search_query}'")
    else:
        # 검색 키워드가 없으면 재구성된 질문 사용
        chunk_search_query = search_query
        print(f"질문 기반 청크 검색 쿼리: '{chunk_search_query}'")

    # 파일이 선정되지 않은 경우 전체 문서에서 검색
    if not relevant_files:
        print("관련 파일이 선정되지 않아 전체 문서에서 검색합니다.")
        document_results = primary_document_db.similarity_search_with_score(chunk_search_query, k=top_k)
    else:
        # 선정된 파일 내에서 관련 청크 검색 (파일당 5개씩)
        print(f"선정된 파일 내에서 파일당 5개씩 청크 검색")

        all_document_results = []

        # 각 파일별로 5개씩 검색
        for file in relevant_files:
            filter_dict = {"filename": {"$eq": file}}
            file_results = primary_document_db.similarity_search_with_score(
                chunk_search_query, k=5, filter=filter_dict
            )

            print(f"파일 '{file}'에서 {len(file_results)}개 청크 검색됨")
            all_document_results.extend(file_results)

        # 전체 결과를 유사도 순으로 정렬하여 상위 top_k개 선택
        document_results = sorted(all_document_results, key=lambda x: x[1])

    print(f"최종 검색된 청크 수: {len(document_results)}")

    # 청크 정보 추출
    contexts = []
    for i, (doc, score) in enumerate(document_results):
        contexts.append({
            "content": doc.page_content,
            "사업명": doc.metadata.get("사업명", ""),
            "발주기관": doc.metadata.get("발주기관", ""),
            "filename": doc.metadata.get("filename", ""),
            "score": score,
            "project_context": "키워드 기반 개선된 검색"
        })

    # 컨텍스트 정보 문자열로 변환 (메타데이터 포함)
    context_text = ""

    # 먼저 선정된 파일의 모든 메타데이터 추가
    if file_metadata_dict:
        context_text += "\n=== 선정된 파일의 메타데이터 ===\n"
        for file, metadata in file_metadata_dict.items():
            context_text += f"\n## 파일: {file} ##\n"

            # 주요 메타데이터 항목 먼저 정렬하여 출력
            priority_fields = ['사업명', '발주 기관', '사업 요약', '사업 금액',
                            '공고 번호', '공고 차수', '공개 일자',
                            '입찰 참여 시작일', '입찰 참여 마감일']

            # 주요 항목 출력
            for field in priority_fields:
                if field in metadata and metadata[field]:
                    context_text += f"{field}: {metadata[field]}\n"

            # 나머지 항목 출력
            for field, value in metadata.items():
                if field not in priority_fields and field != '파일명' and field != '텍스트':
                    context_text += f"{field}: {value}\n"

    # 청크 내용 추가
    context_text += "\n\n=== 문서 내용 ===\n"
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n--- 문서 {i} ---\n"
        context_text += f"사업명: {ctx['사업명']}\n"
        context_text += f"발주기관: {ctx['발주기관']}\n"
        context_text += f"내용: {ctx['content']}\n"

    metadata_preview = context_text.split("=== 문서 내용 ===")[0]  # 메타데이터 부분만 추출

    # 대화 맥락을 고려한 시스템 프롬프트 템플릿
    system_template = """
    당신은 공공입찰 공고 문서를 기반으로 정확한 정보를 제공하는 전문가입니다.
    사용자의 질문에 답하기 위해 제공된 문서 정보와 메타데이터를 활용하세요.

    대화 맥락 정보:
    {conversation_context}

    제공된 문서 정보 및 메타데이터:
    {context}

    사용자는 "{original_query}"라고 질문했으며, 이는 "{reformulated_query}"라고 해석되었습니다.

    **중요한 지시사항:**
    1. 제공된 문서 정보를 바탕으로 질문에 대한 답변을 생성하세요.
    2. 동시에, 제공된 문서 정보가 질문에 답변하기에 충분한지 판단하세요.
    3. 응답은 반드시 다음 JSON 형식으로 작성하세요:

    {{
        "response": "질문에 대한 상세한 답변 내용",
        "is_adequate": true/false
    }}

    **is_adequate 판단 기준:**
    - true: 제공된 문서에 질문과 직접적으로 관련된 구체적인 정보가 있어 의미있는 답변 생성 가능
    - false: 제공된 문서에 질문과 관련된 충분한 정보가 없어 추가 검색이 필요

    **주의사항:**
    - 관련 정보가 부분적으로만 있어도 그 정보로 답변할 수 있다면 is_adequate=true
    - 단순히 주제만 유사하고 구체적인 답변을 할 수 없다면 is_adequate=false
    - JSON 형식을 정확히 지켜주세요
    """

    human_template = "{question}"

    # 프롬프트 템플릿 생성
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 컨텍스트가 비어있는지 확인
    if not contexts and not file_metadata_dict:
        response_content = "해당 정보가 제공된 문서에 없습니다."
    else:
        # 응답 생성 (두 번째 LLM 호출)
        response = chat.invoke(
            chat_prompt.format_messages(
                conversation_context=conversation_context,
                context=context_text,
                original_query=query,
                reformulated_query=search_query,
                question=search_query
            )
        )
        response_content = response.content

    # JSON 파싱 시도
    try:
        # JSON 부분만 추출
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            result = json.loads(json_str)
            primary_result = {
                "response": result.get("response", response_content),
                "is_adequate": result.get("is_adequate", False)
            }
        else:
            # JSON 형식이 아닌 경우 전체를 응답으로 처리하고 적절성은 False로
            primary_result = {"response": response_content, "is_adequate": False}
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        primary_result = {"response": response_content, "is_adequate": False}

    if "is_adequate" in primary_result and primary_result["is_adequate"]:
        # 결과 구성
        result = {
            "query": query,
            "reformulated_query": search_query,
            "relevant_files": relevant_files,
            "contexts": contexts,
            "response": primary_result["response"],
            "file_metadata": file_metadata_dict,
            "metadata_context": metadata_preview,
            "referenced_chunks": [
                {
                    "content": ctx["content"],
                    "사업명": ctx["사업명"],
                    "발주기관": ctx["발주기관"],
                    "similarity_score": ctx["score"],
                    "project_context": ctx.get("project_context", "일반 검색")
                } for ctx in contexts
            ],
            "processing_method": "일반 검색 처리",
            "db_used": "Primary DB"
        }

        # 대화 맥락에 추가
        conversation_manager.add_interaction(query, primary_result["response"], result, search_query)  # search_query가 reformulated_query

        return result

    primary_referenced_chunks = [
                {
                    "content": ctx["content"],
                    "사업명": ctx["사업명"],
                    "발주기관": ctx["발주기관"],
                    "similarity_score": ctx["score"],
                    "project_context": ctx.get("project_context", "일반 검색")
                } for ctx in contexts
            ]

    # 선정된 파일 내에서 관련 청크 검색 (파일당 5개씩)
    print(f"선정된 파일 내에서 파일당 5개씩 청크 검색")

    all_document_results = []

    # 각 파일별로 5개씩 검색
    for file in relevant_files:
        filter_dict = {"filename": {"$eq": file}}
        file_results = secondary_document_db.similarity_search_with_score(
            chunk_search_query, k=5, filter=filter_dict
        )

        print(f"파일 '{file}'에서 {len(file_results)}개 청크 검색됨")
        all_document_results.extend(file_results)

    # 전체 결과를 유사도 순으로 정렬하여 상위 top_k개 선택
    document_results = sorted(all_document_results, key=lambda x: x[1])

    print(f"최종 검색된 청크 수: {len(document_results)}")

    # 청크 정보 추출
    contexts = []
    for i, (doc, score) in enumerate(document_results):
        contexts.append({
            "content": doc.page_content,
            "사업명": doc.metadata.get("사업명", ""),
            "발주기관": doc.metadata.get("발주기관", ""),
            "filename": doc.metadata.get("filename", ""),
            "score": score,
            "project_context": "키워드 기반 개선된 검색"
        })

    # 컨텍스트 정보 문자열로 변환 (메타데이터 포함)
    context_text = ""

    # 먼저 선정된 파일의 모든 메타데이터 추가
    if file_metadata_dict:
        context_text += "\n=== 선정된 파일의 메타데이터 ===\n"
        for file, metadata in file_metadata_dict.items():
            context_text += f"\n## 파일: {file} ##\n"

            # 주요 메타데이터 항목 먼저 정렬하여 출력
            priority_fields = ['사업명', '발주 기관', '사업 요약', '사업 금액',
                            '공고 번호', '공고 차수', '공개 일자',
                            '입찰 참여 시작일', '입찰 참여 마감일']

            # 주요 항목 출력
            for field in priority_fields:
                if field in metadata and metadata[field]:
                    context_text += f"{field}: {metadata[field]}\n"

            # 나머지 항목 출력
            for field, value in metadata.items():
                if field not in priority_fields and field != '파일명' and field != '텍스트':
                    context_text += f"{field}: {value}\n"

    # 청크 내용 추가
    context_text += "\n\n=== 문서 내용 ===\n"
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n--- 문서 {i} ---\n"
        context_text += f"사업명: {ctx['사업명']}\n"
        context_text += f"발주기관: {ctx['발주기관']}\n"
        context_text += f"내용: {ctx['content']}\n"

    metadata_preview = context_text.split("=== 문서 내용 ===")[0]  # 메타데이터 부분만 추출

    # 대화 맥락을 고려한 시스템 프롬프트 템플릿
    system_template = """
    당신은 공공입찰 공고 문서를 기반으로 정확한 정보를 제공하는 전문가입니다.
    사용자의 질문에 답하기 위해 제공된 문서 정보와 메타데이터를 활용하세요.

    대화 맥락 정보:
    {conversation_context}

    제공된 문서 정보 및 메타데이터:
    {context}

    사용자는 "{original_query}"라고 질문했으며, 이는 "{reformulated_query}"라고 해석되었습니다.

    제공된 문서 정보와 메타데이터에 근거하여 명확하고 정확한 답변을 제공하세요.
    문서 정보에 관련 내용이 없으면 "해당 정보가 제공된 문서에 없습니다."라고 답변하세요.
    관련 정보가 부분적으로 있으면 있는 정보만을 제공하세요.
    메타데이터 정보(사업 금액, 공고 번호, 입찰 참여 기간 등)가 질문과 관련이 있으면 답변에 포함하세요.

    이전 대화에서 언급된 내용을 고려하여 답변하되, 현재 질문에 집중해서 답변하세요.
    사용자가 이전에 언급한 특정 사업이나 기관에 대해 질문하고 있다면, 그 맥락을 유지하여 답변하세요.
    """

    human_template = "{question}"

    # 프롬프트 템플릿 생성
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 컨텍스트가 비어있는지 확인
    if not contexts and not file_metadata_dict:
        response_content = "해당 정보가 제공된 문서에 없습니다."
    else:
        # 응답 생성 (두 번째 LLM 호출)
        response = chat.invoke(
            chat_prompt.format_messages(
                conversation_context=conversation_context,
                context=context_text,
                original_query=query,
                reformulated_query=search_query,
                question=search_query
            )
        )
        response_content = response.content

    # 결과 구성
    result = {
        "query": query,
        "reformulated_query": search_query,
        "relevant_files": relevant_files,
        "contexts": contexts,
        "response": response_content,
        "file_metadata": file_metadata_dict,
        "metadata_context": metadata_preview,
        "referenced_chunks": [
            {
                "content": ctx["content"],
                "사업명": ctx["사업명"],
                "발주기관": ctx["발주기관"],
                "similarity_score": ctx["score"],
                "project_context": ctx.get("project_context", "일반 검색")
            } for ctx in contexts
        ],
        "primary_referenced_chunks": primary_referenced_chunks,
        "processing_method": "일반 검색 처리",
        "db_used": "Secondary DB"
    }

    total_response_content = f'\n첫번째 응답 : {primary_result["response"]}  \n두번째 응답 : {response_content}'

    # 대화 맥락에 추가
    conversation_manager.add_interaction(query, total_response_content, result, search_query)  # search_query가 reformulated_query

    return result