import json
import pandas as pd
from utils.text_utils import clean_filename

def load_metadata(metadata_path):
    """메타데이터 CSV 파일 로드 및 처리"""
    metadata_df = pd.read_csv(metadata_path)
    # 필요한 컬럼: 공고 번호, 공고 차수, 사업명, 사업 금액, 발주 기관, 공개 일자, 입찰 참여 시작일, 입찰 참여 마감일, 사업 요약, 파일형식, 파일명, 텍스트
    metadata_df['파일명'] = metadata_df['파일명'].apply(clean_filename)
    print(f"메타데이터 로드 완료: {metadata_df.shape[0]}개의 공고 데이터")
    return metadata_df

def load_documents(file_path):
    """JSON 파일에서 문서 데이터 로드 및 전처리 (새로운 페이지 기반 구조)"""
    # JSON 로드
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 청크 정보를 저장할 리스트 (각 페이지가 하나의 청크)
    chunks = []

    if isinstance(data, list):
        for file_data in data:  # 각 파일별 데이터
            if isinstance(file_data, list):
                for page_data in file_data:  # 각 페이지별 데이터
                    try:
                        filename = clean_filename(page_data.get("filename", ""))
                        page_number = page_data.get("page_number", 0)
                        merged_page_content = page_data.get("merged_page_content", "").strip()

                        # 내용이 있는 페이지만 처리
                        if merged_page_content:
                            chunks.append({
                                "filename": filename,
                                "page_number": page_number,
                                "content": merged_page_content
                            })
                    except Exception as e:
                        print(f"❗ 페이지 처리 중 오류 발생: {e}")

    print(f"문서 로드 완료: {len(chunks)}개의 청크 (페이지 단위)")
    return chunks