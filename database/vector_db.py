import numpy as np
from langchain.vectorstores import Chroma
from langchain.schema import Document

def create_document_db(chunks, metadata_df, embeddings, db_name):
    """문서 청크 기반 벡터 DB 생성 (페이지 단위 청크)"""

    all_documents = []

    # 메타데이터를 파일명으로 인덱싱하여 빠르게 접근할 수 있도록 함
    metadata_dict = {}
    for idx, row in metadata_df.iterrows():
        file_name = row['파일명'].replace('.pdf', '').strip() if '파일명' in row else ""
        metadata_dict[file_name] = row.to_dict()

    # 각 청크(페이지)별로 Document 객체 생성
    for chunk_info in chunks:
        filename = chunk_info["filename"]
        page_number = chunk_info["page_number"]
        content = chunk_info["content"]

        # 메타데이터 매핑
        file_metadata = metadata_dict.get(filename, {})

        # 메타데이터 추출
        project_name = file_metadata.get('사업명', "")
        agency = file_metadata.get('발주 기관', "")

        # Document 객체 생성
        doc = Document(
            page_content=content,
            metadata={
                "filename": filename,
                "page_number": page_number,
                "사업명": project_name,
                "발주기관": agency,
                "공고번호": file_metadata.get('공고 번호', ""),
                "공고차수": file_metadata.get('공고 차수', ""),
                "사업금액": file_metadata.get('사업 금액', ""),
                "공개일자": file_metadata.get('공개 일자', ""),
                "입찰시작일": file_metadata.get('입찰 참여 시작일', ""),
                "입찰마감일": file_metadata.get('입찰 참여 마감일', ""),
                "사업요약": file_metadata.get('사업 요약', "")
            }
        )
        all_documents.append(doc)

    # 청크 길이 통계 출력
    chunk_lengths = [len(doc.page_content) for doc in all_documents]
    print(f"청크 길이 분포 (총 {len(chunk_lengths)}개):")
    print(f"- 최소 길이: {min(chunk_lengths)}")
    print(f"- 최대 길이: {max(chunk_lengths)}")
    print(f"- 평균 길이: {np.mean(chunk_lengths):.2f}")
    print(f"- 중간값: {np.median(chunk_lengths):.2f}")

    # 청크 길이 히스토그램 출력 (구간별 분포)
    bins = [0, 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2500, 3000]
    hist, bin_edges = np.histogram(chunk_lengths, bins=bins)
    for i in range(len(hist)):
        print(f"- {bin_edges[i]}-{bin_edges[i+1]} 자: {hist[i]}개 ({hist[i]/len(chunk_lengths)*100:.1f}%)")

    # Chroma DB 생성
    document_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory= "./" + db_name
    )

    print(f"문서 벡터 DB 생성 완료: {len(all_documents)}개의 청크")
    return document_db