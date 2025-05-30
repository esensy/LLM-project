from langchain.vectorstores import Chroma
from langchain.schema import Document

def create_metadata_db(metadata_df, embeddings):
    """메타데이터 기반 벡터 DB 생성 (사업명 + 발주기관 기반)"""

    documents = []

    for idx, row in metadata_df.iterrows():
        # 사업명과 발주기관을 함께 사용하여 임베딩
        project_name = row['사업명'] if '사업명' in row else ""
        agency = row['발주 기관'] if '발주 기관' in row else ""
        file_name = row['파일명'].replace('.pdf', '').strip() if '파일명' in row else ""

        # 풍부한 컨텍스트로 임베딩 - 사업명(2번 반복) + 발주기관 + 핵심 키워드
        # 사업명을 2번 반복하여 가중치를 높임(일단 생략)
        enriched_content = f"{project_name} {agency}"

        # 키워드 보강 - 프로젝트명에 시스템, 구축, 용역 등의 키워드가 있으면 추가
        keywords = ["시스템", "사업", "구축", "개발", "용역", "플랫폼", "고도화", "기능개선"]
        for keyword in keywords:
            if keyword in project_name:
                enriched_content += f" {keyword}"

        # print(f"사업명: {project_name}")
        # print(f"발주기관: {agency}")
        # print(f"임베딩 컨텐츠: {enriched_content}")

        # Document 객체 생성
        doc = Document(
            page_content=enriched_content,  # 풍부한 컨텍스트로 변경
            metadata={
                "사업명": project_name,
                "공고번호": row['공고 번호'] if '공고 번호' in row else "",
                "공고차수": row['공고 차수'] if '공고 차수' in row else "",
                "발주기관": agency,
                "사업금액": row['사업 금액'] if '사업 금액' in row else "",
                "공개일자": row['공개 일자'] if '공개 일자' in row else "",
                "입찰시작일": row['입찰 참여 시작일'] if '입찰 참여 시작일' in row else "",
                "입찰마감일": row['입찰 참여 마감일'] if '입찰 참여 마감일' in row else "",
                "사업요약": row['사업 요약'] if '사업 요약' in row else "",
                "파일명": file_name
            }
        )
        documents.append(doc)

    # Chroma DB 생성
    metadata_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_metadata_db"
    )

    print(f"메타데이터 벡터 DB 생성 완료: {len(documents)}개의 문서")
    return metadata_db