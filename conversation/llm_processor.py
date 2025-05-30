# 통합된 LLM 기반 대화 맥락 파악 및 메타데이터 질의 처리 함수
def process_with_llm(query, conversation_context, metadata_df, chat):
    """
    LLM을 이용해 대화 맥락을 파악하고 메타데이터 기반 질의 처리를 통합하여 수행

    Args:
        query (str): 사용자 질문
        conversation_context (str): 대화 컨텍스트
        metadata_df (DataFrame): 메타데이터 데이터프레임
        chat: ChatOpenAI 객체

    Returns:
        dict: 처리 결과를 담은 딕셔너리
    """
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    import json
    import re

    # 메타데이터 컬럼 정보 추출
    columns = metadata_df.columns.tolist()
    column_samples = {}
    for col in columns:
        if col in metadata_df.columns:
            sample_values = metadata_df[col].dropna().sample(min(3, len(metadata_df[col].dropna()))).tolist()
            column_samples[col] = sample_values

    # 시스템 프롬프트 템플릿 - 대화 맥락 파악 및 메타데이터 처리 통합
    system_template = """
    당신은 공공입찰 공고 문서 검색 시스템의 질문 분석기 및 메타데이터 처리기입니다.
    두 가지 주요 기능을 수행해야 합니다:

    1. 사용자 질문 및 대화 맥락 분석:
    - 질문이 메타데이터만으로 답변 가능한지 여부 판단
    - 대화 맥락을 고려한 질문 재구성
    - 질문에서 사업명과 발주기관을 페어로 추출 (관련있는 것끼리 묶어서)
    - 중요: 특정 사업명이 질문에 포함된 경우 문서 내용 검색이 필요합니다 (metadata_only=false)

    2. 메타데이터 기반 질의 처리 (메타데이터만으로 처리 가능한 경우):
    - 추출된 사업명-발주기관 페어에 기반하여 적절한 메타데이터 검색
    - 검색 결과를 바탕으로 응답 생성

    제공된 메타데이터 컬럼:
    {columns}

    메타데이터 컬럼 샘플 값:
    {column_samples}

    이전 대화 컨텍스트:
    {conversation_context}

    **대화 맥락 고려 및 질문 재구성 핵심 규칙:**

    1. **직전 질문 우선 원칙**:
    - 현재 질문이 이전 대화와 연관된 경우, 가장 최근(직전) 질문을 우선 참조
    - "그럼", "그", "해당", "위" 등의 지시어가 있으면 직전 질문과 연결

    2. **새로운 검색 범위 인식**:
    - "다른", "다른 기관", "다른 사업", "또 다른", "별도의" 등의 표현이 있으면 **새로운 검색 범위**를 의미
    - 이 경우 이전 사업명을 포함시키지 않고, 새로운 검색 범위로 질문 재구성
    - 예: "다른 기관이 발주한 교육 관련 사업" → 기존 사업명 제외하고 검색

    3. **동일 사업 후속 질문 인식** (지시어 기반):
    - 지시어("그", "그럼", "해당") + 구체적 내용 질문 = 직전 언급 사업에 대한 후속 질문
    - 이 경우 직전 질문에서 언급된 구체적 사업명 포함
    - 예: "그럼 모니터링 업무는?" → 직전 질문의 사업명 포함

    4. **복수 사업 참조 질문 인식** ("각각" 표현):
    - "각각", "둘 다", "모두" 등의 표현 + 이전에 언급된 동일 기관명 = 이전 구체적 사업들 참조
    - 이 경우 이전 대화에서 언급된 해당 기관들의 구체적 사업명을 포함
    - 예: "고려대학교랑 광주과학기술원 각각 응답시간 요구사항" → 이전에 언급된 두 기관의 구체적 사업명 포함

    5. **맥락 연결 우선순위**:
    - 1순위: 직전 질문 지시어 연결 ("그럼", "그" 등)
    - 2순위: 복수 사업 참조 ("각각", "둘 다" + 동일 기관명)
    - 3순위: 현재 질문에 명시된 기관/사업과 일치하는 이전 질문
    - 단, "다른/별도" 표현이 있으면 맥락 연결 하지 않음

    **메타데이터 질문 표준화 규칙 (규칙 기반 처리를 위한 필수 사항):**

    메타데이터 관련 질문들은 반드시 표준화된 표현으로 재구성해야 합니다:

    1. **사업 금액 관련 표준화**:
    - 다양한 표현 → 표준 표현
    - "가장 많은 자금", "예산 규모가 최대", "고액 투자", "최다 예산" → "사업 금액이 가장 큰"
    - "가장 적은 예산", "최소 비용", "저액", "소규모" → "사업 금액이 가장 작은"
    - "N억 이상", "N만원 초과", "고액" → "사업 금액이 N원 이상/초과"

    2. **발주 기관 관련 표준화**:
    - "어떤 기관에서", "누가 발주한", "발주처", "주관 기관" → "발주 기관"
    - "모든 조직", "전체 기관", "발주처 목록" → "모든 발주 기관"

    3. **날짜 관련 표준화**:
    - "최신", "가장 최근", "방금 나온", "새로운" → "가장 최근에 공개된"
    - "오래된", "예전", "초기", "처음" → "가장 먼저 공개된"

    4. **공고 번호 관련 표준화**:
    - "번호가 XXX인", "공고번호 XXX" → "공고 번호가 XXX인"

    엔티티 추출 및 질문 재구성 규칙:
    - 사업명과 발주기관이 함께 언급되면 페어로 묶어서 저장
    - 사업명만 언급된 경우: 발주기관은 빈 문자열로 설정
    - 발주기관만 언급된 경우: 사업명은 빈 문자열로 설정

    rule_based_query 판단 기준:
    - 다음과 같은 질문들은 규칙 기반 처리가 가능합니다:
    * 공고 번호로 검색하는 질문
    * 사업 금액 관련 질문 (최대, 최소, 범위 등 - 표준화된 표현으로 재구성 필요)
    * 발주 기관별 사업 조회 (표준화된 표현으로 재구성 필요)
    * 날짜 관련 질문 (최근, 특정 기간 등 - 표준화된 표현으로 재구성 필요)
    * 단순한 메타데이터 필드 조회

    - 다음과 같은 질문들은 LLM 메타데이터 처리가 필요합니다:
    * 복잡한 조건 조합이 필요한 질문
    * 자연어 표현이 복잡한 질문
    * 추론이나 해석이 필요한 질문

    분석 시 다음 상황을 구분하세요:
    1. 특정 사업명 포함 & 상세내용 요청 → metadata_only=false (문서 검색 필요)
    2. 단순한 메타데이터 질의 → metadata_only=true & rule_based_query=true (규칙 기반 처리 가능)
    3. 복잡한 메타데이터 질의 → metadata_only=true & rule_based_query=false (LLM 메타데이터 처리)
    4. 일반 검색 질의 → metadata_only=false (문서 검색 필요)

    단계적 응답 제공:
    1. 분석 결과: 메타데이터로만 처리 가능 여부, 재구성된 질문, 추출된 엔티티 등
    2. 메타데이터로 처리 가능한 경우, 해당 질문에 대한 응답 생성
    """

    # 인간 프롬프트 템플릿
    human_template = """
    사용자 질문: {query}

    위 질문을 분석하여 JSON 형식으로 답변해주세요. 반드시 유효한 JSON 형식이어야 하며, 다음 형식을 정확히 따라야 합니다.

    {{
    "analysis": {{
        "metadata_only": true,
        "rule_based_query": false,
        "reformulated_query": "대화 맥락과 표준화 규칙을 모두 고려하여 재구성된 질문",
        "search_keywords": ["질문에서 추출한 핵심 키워드1", "질문에서 추출한 핵심 키워드2"],
        "relevant_columns": ["관련 컬럼1", "관련 컬럼2"],
        "entities": {{
            "project_agency_pairs": [
                {{"사업명": "구체적한 사업명1", "발주기관": "발주기관1"}},
                {{"사업명": "구체적한 사업명2", "발주기관": "발주기관2"}}
            ],
            "금액": ["언급된 금액"],
            "날짜": ["언급된 날짜"]
        }},
        "reasoning": "대화 맥락 분석 과정과 표준화 적용 과정을 포함한 재구성 근거를 상세히 설명"
    }},
    "metadata_response": ""
    }}

    **맥락 연결 판단 로직** (반드시 순서대로 체크):
    1. 현재 질문에 이전 대화에서 언급된 동일한 기관명이 있는가?
    2. "각각", "둘 다" 등의 복수 참조 표현이 있는가?
    3. "다른", "별도" 등의 새로운 범위 키워드가 없는가?
    → 모두 YES면 이전에 언급된 구체적 사업명들을 포함하여 재구성

    **맥락 연결 방지 키워드**: "다른", "다른 기관", "다른 사업", "또 다른", "별도의", "새로운", "추가로"
    **맥락 연결 지시어**: "그", "그럼", "해당", "위", "앞서", "이전의"
    **복수 사업 참조 표현**: "각각", "둘 다", "모두", "전부"

    **중요한 대화 맥락 판단 예시:**

    케이스 1 - 새로운 검색 범위 (다른 사업 찾기):
    - 이전: "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 알려줘"
    - 현재: "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?"
    - 맥락 연결 판단: 1) 동일 기관명 없음 2) 복수 참조 표현 없음 3) "다른 기관" 키워드 있음 → NO
    - project_agency_pairs: [] (기존 사업명 포함하지 않음)
    - reformulated_query: "교육이나 학습 관련해서 다른 기관이 발주한 사업이 있는지 알려 주세요"

    케이스 2 - 동일 사업 후속 질문 (지시어 있음):
    - 이전: "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?"
    - 현재: "그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘"
    - 판단: "그럼"이라는 지시어로 직전 질문의 동일 사업에 대한 후속 질문 (맥락 연결 로직보다 지시어 우선)
    - project_agency_pairs: [{{"사업명": "극저온시스템 운전 용역", "발주기관": "기초과학연구원"}}]
    - reformulated_query: "기초과학연구원이 발주한 극저온시스템 운전 용역 사업에서 모니터링 업무에 대한 요청사항이 있는지 알려 주세요"

    케이스 3 - 복수 사업 참조 질문 ("각각" 표현):
    - 이전: "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?"
    - 현재: "고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나?"
    - 맥락 연결 판단: 1) 동일 기관명 있음 (고려대학교, 광주과학기술원) 2) 복수 참조 표현 있음 ("각각") 3) 새로운 범위 키워드 없음 → YES
    - project_agency_pairs: [{{"사업명": "차세대 포털 시스템", "발주기관": "고려대학교"}}, {{"사업명": "학사 시스템 기능개선", "발주기관": "광주과학기술원"}}]
    - reformulated_query: "고려대학교 차세대 포털 시스템 사업과 광주과학기술원 학사 시스템 기능개선 사업에서 각각 응답 시간에 대한 요구사항이 있는지 알려 주세요"

    케이스 4 - 진짜 맥락 없는 독립 질문:
    - 현재: "서울대학교와 연세대학교 발주 사업 중 예산이 큰 것은?"
    - 판단: 이전 대화에서 해당 기관들이 언급되지 않은 완전히 새로운 질문
    - project_agency_pairs: [{{"사업명": "", "발주기관": "서울대학교"}}, {{"사업명": "", "발주기관": "연세대학교"}}]
    - reformulated_query: "서울대학교와 연세대학교가 발주한 사업 중 예산이 큰 것이 무엇인지 알려 주세요"

    **메타데이터 질문 표준화 예시:**

    표준화 케이스 1 - 사업 금액 관련:
    - 질문: "가장 많은 자금이 배정된 항목은 뭔가요?"
    - reformulated_query: "사업 금액이 가장 큰 사업은 무엇인가요?"
    - reasoning: "자금 배정 관련 질문을 규칙 기반 처리가 가능한 표준화된 사업 금액 표현으로 재구성"

    표준화 케이스 2 - 발주 기관 관련:
    - 질문: "어떤 조직에서 교육 사업을 주관했나요?"
    - reformulated_query: "교육 사업을 발주한 기관은 어디인가요?"
    - reasoning: "조직/주관 표현을 표준화된 발주 기관 표현으로 재구성"

    표준화 케이스 3 - 날짜 관련:
    - 질문: "최신으로 발표된 공고가 뭐야?"
    - reformulated_query: "가장 최근에 공개된 공고는 무엇인가요?"
    - reasoning: "최신/발표 표현을 표준화된 공개 일자 표현으로 재구성"

    entities 추출 및 재구성 규칙:
    1. 새로운 검색 범위 질문 → project_agency_pairs에 기존 사업명 포함하지 않음
    2. 동일 사업 후속 질문 (지시어) → 직전 질문의 구체적 사업명 포함
    3. 복수 사업 참조 질문 ("각각") → 이전에 언급된 해당 기관들의 구체적 사업명 포함
    4. 독립적인 질문 → 현재 질문 내용만 반영
    5. 메타데이터 관련 질문 → 반드시 표준화된 표현으로 재구성

    search_keywords 추출 규칙:
    1. 반드시 사용자 질문에 실제로 포함된 단어들만 추출
    2. 발주기관명과 사업명은 제외 (entities에서 별도 관리)
    3. 문서 내용 검색에 유용한 명사, 동사, 형용사만 포함
    4. 조사, 접속사, 대명사 등은 제외
    5. 유사어나 동의어를 임의로 추가하지 말고 원문 그대로 사용
    """

    # 프롬프트 템플릿 생성
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 프롬프트 포맷팅 및 GPT 호출
    formatted_prompt = chat_prompt.format_messages(
        columns=columns,
        column_samples=column_samples,
        conversation_context=conversation_context,
        query=query
    )

    response = chat.invoke(formatted_prompt)
    response_content = response.content

    print("GPT 응답:")
    print(response_content)

    # JSON 추출 및 파싱
    try:
        # 우선 ```json과 ``` 사이의 내용 추출 시도
        json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 실패하면 직접 중괄호로 둘러싸인 JSON 찾기 시도
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # 그래도 실패하면 전체 응답 사용
                json_str = response_content

        # 불필요한 공백과 개행 제거
        json_str = json_str.strip()

        # JSON 파싱
        result = json.loads(json_str)

        # 기본 구조 확인 및 설정
        if "analysis" not in result:
            result["analysis"] = {}

        analysis = result["analysis"]

        # 필수 키가 있는지 확인하고 없으면 기본값 설정
        if "metadata_only" not in analysis:
            analysis["metadata_only"] = False
        if "rule_based_query" not in analysis:
            analysis["rule_based_query"] = False
        if "reformulated_query" not in analysis:
            analysis["reformulated_query"] = query
        if "relevant_columns" not in analysis:
            analysis["relevant_columns"] = []
        if "entities" not in analysis:
            analysis["entities"] = {"project_agency_pairs": [], "금액": [], "날짜": []}
        elif "project_agency_pairs" not in analysis["entities"]:
            analysis["entities"]["project_agency_pairs"] = []
        if "reasoning" not in analysis:
            analysis["reasoning"] = "분석 실패"
        if "search_keywords" not in analysis:
            analysis["search_keywords"] = []
        if "metadata_response" not in result:
            result["metadata_response"] = ""

        return result

    except Exception as e:
        print(f"처리 실패: {type(e).__name__}: {e}")
        # 오류 발생 시 기본값 반환
        return {
            "analysis": {
                "metadata_only": False,
                "reformulated_query": query,
                "relevant_columns": [],
                "entities": {
                    "사업명": [],
                    "발주기관": [],
                    "금액": [],
                    "날짜": []
                },
                "reasoning": f"오류 발생: {str(e)}"
            },
            "metadata_response": ""
        }