import pandas as pd
import re

def extract_numeric_amount(amount_str):
    """금액 문자열에서 숫자만 추출하고 단위에 따라 변환"""
    if pd.isna(amount_str) or not amount_str:
        return 0

    amount_str = str(amount_str).replace(',', '')

    # 숫자와 단위 추출
    unit_map = {
        '억': 100000000,
        '천만': 10000000,
        '백만': 1000000,
        '만': 10000,
        '천': 1000
    }

    # 기본 숫자 추출
    numbers = re.findall(r'\d+', amount_str)
    if not numbers:
        return 0

    # 먼저 단순 숫자만 결합
    numeric_value = ''.join(numbers)

    # 단위가 있는지 확인
    for unit, multiplier in unit_map.items():
        if unit in amount_str:
            # 단위가 있으면 해당 단위로 변환
            base_value = int(numeric_value)
            return base_value * multiplier

    # 단위가 없으면 그대로 반환
    return int(numeric_value)

def preprocess_metadata(df):
    """메타데이터 데이터프레임 전처리"""
    # 날짜 필드 변환
    date_fields = ['공개 일자', '입찰 참여 시작일', '입찰 참여 마감일']
    for field in date_fields:
        if field in df.columns:
            df[f'{field}_datetime'] = pd.to_datetime(df[field], errors='coerce')

    # 금액 필드 변환
    if '사업 금액' in df.columns:
        df['사업 금액_숫자'] = df['사업 금액'].apply(extract_numeric_amount)

    return df