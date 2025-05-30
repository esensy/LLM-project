# LLM 호출을 최적화한 대화 맥락 관리 및 질문 처리 클래스
class ConversationManager:
    def __init__(self, max_history=5, metadata_df=None):
        """
        대화 맥락을 관리하는 클래스
        """
        self.history = []
        self.max_history = max_history
        self.mentioned_entities = {
            "사업명": [],
            "발주기관": [],
            "금액": [],
            "파일명": []
        }
        # 질문-응답 로그 (질문, 재구성된 질문, 응답)
        self.qa_logs = []
        # 메타데이터 DataFrame 참조 추가
        self._metadata_df = metadata_df

    def add_interaction(self, query, response, metadata=None, reformulated_query=None):
        """
        대화 기록에 새로운 상호작용 추가

        Args:
            query (str): 사용자 질문
            response (str): 시스템 응답
            metadata (dict, optional): 관련 메타데이터
            reformulated_query (str, optional): 재구성된 질문
        """
        # 기록 추가
        interaction = {
            "query": query,
            "response": response,
            "metadata": metadata or {}
        }

        self.history.append(interaction)

        # 최대 히스토리 유지
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # 언급된 엔티티 추출 및 저장
        self._extract_entities(interaction)

        # QA 로그에 추가
        self.qa_logs.append({
            "query": query,
            "reformulated_query": reformulated_query if reformulated_query and reformulated_query != query else None,
            "response": response
        })

    def _extract_entities(self, interaction):
        """
        대화에서 언급된 엔티티(사업명, 발주기관 등) 추출하여 저장
        """
        # 메타데이터에서 엔티티 추출
        metadata = interaction.get("metadata", {})

        # 여러 가지 메타데이터 구조 처리
        file_metadata = {}

        # 방법 1: file_metadata 키가 있는 경우 (일반 검색, 하이브리드 검색)
        if "file_metadata" in metadata:
            file_metadata = metadata.get("file_metadata", {})

        # 방법 2: 직접 파일 정보가 있는 경우 (규칙 기반 처리)
        elif "relevant_files" in metadata and hasattr(self, '_last_metadata_df'):
            # relevant_files에서 파일명을 가져와서 메타데이터 조회
            relevant_files = metadata.get("relevant_files", [])
            for file in relevant_files:
                # 글로벌 메타데이터에서 해당 파일 정보 찾기
                if hasattr(self, '_metadata_df'):
                    file_rows = self._metadata_df[self._metadata_df['파일명'] == file]
                    if not file_rows.empty:
                        file_metadata[file] = file_rows.iloc[0].to_dict()

        # 방법 3: contexts에서 추출 (문서 검색 결과)
        contexts = metadata.get("contexts", [])
        for ctx in contexts:
            filename = ctx.get("filename", "")
            if filename:
                # 임시 파일 메타데이터 생성
                temp_metadata = {
                    "사업명": ctx.get("사업명", ""),
                    "발주 기관": ctx.get("발주기관", ""),
                    "파일명": filename
                }
                file_metadata[filename] = temp_metadata

        # 엔티티 추출 및 저장
        for file_info in file_metadata.values():
            project_name = file_info.get("사업명")
            if project_name and project_name not in self.mentioned_entities["사업명"]:
                self.mentioned_entities["사업명"].append(project_name)

            agency = file_info.get("발주 기관")
            if agency and agency not in self.mentioned_entities["발주기관"]:
                self.mentioned_entities["발주기관"].append(agency)

            file_name = file_info.get("파일명")
            if file_name and file_name not in self.mentioned_entities["파일명"]:
                self.mentioned_entities["파일명"].append(file_name)

    def get_conversation_context(self, num_recent=3):
        """
        최근 대화 컨텍스트 반환

        Args:
            num_recent (int): 가져올 최근 대화 수

        Returns:
            str: 포맷팅된 대화 컨텍스트
        """
        if not self.history:
            return "이전 대화 없음"

        # 최근 대화만 가져오기
        recent_history = self.history[-num_recent:] if len(self.history) > num_recent else self.history

        # 대화 컨텍스트 생성
        context = "=== 이전 대화 내용 ===\n"
        for i, interaction in enumerate(recent_history):
            context += f"사용자: {interaction['query']}\n"
            context += f"시스템: {interaction['response']}\n\n"

        # 최근 언급된 엔티티 추가
        context += "=== 언급된 주요 엔티티 ===\n"
        for entity_type, entities in self.mentioned_entities.items():
            if entities:
                context += f"{entity_type}: {', '.join(entities[-3:])}\n"

        return context