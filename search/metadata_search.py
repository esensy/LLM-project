import re
import pandas as pd

def process_metadata_query(query, df):
    """규칙 기반 메타데이터 질문 처리"""
    # 공고 번호 검색 (최우선)
    번호_match = re.search(r'공고\s*번호\s*(?:가|이)?\s*(\d{5,})', query) or re.search(r'(\d{8,})', query)
    if 번호_match:
        번호 = 번호_match.group(1)
        filtered = df[df['공고 번호'].astype(str).str.contains(번호, na=False)]
        return filtered, f"공고 번호: {번호}"

    # 사업 금액 관련 조건 (패턴 대폭 확장)
    money_keywords = [
        '사업 금액', '예산', '비용', '자금', '예산 규모', '금액',
        '투자', '투자금', '사업비', '총액', '총 비용', '총 예산',
        '고비용', '저비용', '대규모', '소규모'
    ]

    if any(keyword in query for keyword in money_keywords):
        # 최대/최소 금액 질문 (패턴 확장)
        max_keywords = [
            '가장 큰', '가장 많은', '최대', '최고', '제일 큰', '제일 많은',
            '가장 높은', '최고액', '최대액', '가장 비싼', '고액', '대규모',
            '가장 많이 배정', '최다 예산', '최고 수준'
        ]

        min_keywords = [
            '가장 작은', '가장 적은', '최소', '최저', '제일 작은', '제일 적은',
            '가장 낮은', '최소액', '최저액', '가장 싼', '저액', '소규모',
            '가장 적게 배정', '최소 예산', '최저 수준'
        ]

        if any(keyword in query for keyword in max_keywords):
            result = df.sort_values(by='사업 금액_숫자', ascending=False).head(3)
            return result, "사업 금액 내림차순"

        elif any(keyword in query for keyword in min_keywords):
            # 0원은 제외 (금액 정보가 없는 경우)
            non_zero = df[df['사업 금액_숫자'] > 0]
            result = non_zero.sort_values(by='사업 금액_숫자', ascending=True).head(3)
            return result, "사업 금액 오름차순"

        # 금액 범위 질문
        금액_match = re.search(r'(\d[\d,]*)(?:\s*만|\s*천만|\s*억)?(?:\s*원)?(?:\s*이상|\s*이하|\s*초과|\s*미만)', query)
        if 금액_match:
            full_match = 금액_match.group(0)
            raw_amount = 금액_match.group(1).replace(',', '')
            amount = int(raw_amount)

            # 단위 변환
            if '억' in full_match:
                amount *= 100000000
            elif '천만' in full_match:
                amount *= 10000000
            elif '만' in full_match:
                amount *= 10000

            # 비교 조건
            if '이상' in full_match:
                result = df[df['사업 금액_숫자'] >= amount]
                return result, f"사업 금액 {amount}원 이상"
            elif '초과' in full_match:
                result = df[df['사업 금액_숫자'] > amount]
                return result, f"사업 금액 {amount}원 초과"
            elif '이하' in full_match:
                result = df[df['사업 금액_숫자'] <= amount]
                return result, f"사업 금액 {amount}원 이하"
            elif '미만' in full_match:
                result = df[df['사업 금액_숫자'] < amount]
                return result, f"사업 금액 {amount}원 미만"

    # 날짜 관련 질문 (패턴 확장)
    date_keywords = ['공개', '공개 일자', '발표', '게시', '공고']
    recent_keywords = ['최근', '가장 최근', '최신', '새로운', '방금', '최근에']
    old_keywords = ['먼저', '처음', '가장 먼저', '오래된', '예전', '초기']

    if any(keyword in query for keyword in date_keywords):
        if any(keyword in query for keyword in recent_keywords):
            result = df.sort_values(by='공개 일자_datetime', ascending=False).head(3)
            return result, "최근 공개 일자 순"
        elif any(keyword in query for keyword in old_keywords):
            result = df.sort_values(by='공개 일자_datetime', ascending=True).head(3)
            return result, "오래된 공개 일자 순"

    # 입찰 일자 관련 질문
    if '입찰' in query:
        # 입찰 시작일 질문
        if '시작일' in query or '시작하는' in query:
            년도_match = re.search(r'(\d{4})년', query)
            if 년도_match:
                year = 년도_match.group(1)
                result = df[df['입찰 참여 시작일_datetime'] >= f'{year}-01-01']
                return result, f"{year}년 이후 입찰 시작 사업"

        # 입찰 마감일 질문
        if '마감일' in query or '마감되는' in query:
            년월_match = re.search(r'(\d{4})년\s*(\d{1,2})월', query)
            if 년월_match:
                year, month = 년월_match.groups()
                month_int = int(month)

                # 월이 유효한지 확인
                if 1 <= month_int <= 12:
                    month_padded = str(month_int).zfill(2)
                    next_month_int = (month_int % 12) + 1
                    next_year = int(year) + (1 if month_int == 12 else 0)
                    next_month_padded = str(next_month_int).zfill(2)

                    start_date = f'{year}-{month_padded}-01'
                    end_date = f'{next_year}-{next_month_padded}-01'

                    mask = (df['입찰 참여 마감일_datetime'] >= start_date) & (df['입찰 참여 마감일_datetime'] < end_date)
                    result = df[mask]
                    return result, f"{year}년 {month}월 입찰 마감 사업"

    # 처리할 수 없는 메타데이터 질문
    return None

def format_metadata_results(filtered_data, filter_description):
    """필터링된 메타데이터 결과를 포맷팅"""
    context = f"\n=== 메타데이터 검색 결과 ({filter_description}) ===\n"

    if len(filtered_data) > 10:
        context += f"\n총 {len(filtered_data)}개 결과 중 10개 표시\n"

    # 주요 필드 정의
    priority_fields = ['사업명', '발주 기관', '사업 요약', '사업 금액', '공고 번호',
                       '공고 차수', '공개 일자', '입찰 참여 시작일', '입찰 참여 마감일', '파일명']

    # 각 결과에 대한 정보 추가
    for idx, row in filtered_data.iterrows():
        context += f"\n## 결과 {idx+1} ##\n"

        # 주요 필드 우선 출력
        for field in priority_fields:
            if field in row and not pd.isna(row[field]) and row[field]:
                context += f"{field}: {row[field]}\n"

    return context