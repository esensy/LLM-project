import json, re, unicodedata
import pandas as pd 

def print_divider(title = "구분선", color_code = 31):
  print(f"\033[{color_code}m" + "=" * 100 + f" {title} " + "=" * 100 + "\033[0m")

def clean_filename(name: str) -> str:
  """
  파일 이름에서 확장자를 제거하고 유니코드 정규화 (NFC)를 수행.

  Args:
    name: 클리닝할 파일 이름 문자열.

  Returns:
    확장자가 제거되고 정규화된 파일 이름 문자열.
  """
  return re.sub(r"\.(pdf|hwp|docx|xlsx|json)$", "", unicodedata.normalize("NFC", name.strip()), flags = re.IGNORECASE)

# 메타데이터 처리 함수
def load_metadata(metadata_path):
  """메타데이터 CSV 파일 로드 및 처리"""
  print(f"메타데이터 로드 중: {metadata_path}....")
  try:
    metadata_df = pd.read_csv(metadata_path)
    # 필요한 컬럼: 공고 번호, 공고 차수, 사업명, 사업 금액, 발주 기관, 공개 일자, 입찰 참여 시작일, 입찰 참여 마감일, 사업 요약, 파일형식, 파일명, 텍스트.
    if "파일명" in metadata_df.columns:
      metadata_df["파일명"] = metadata_df["파일명"].apply(clean_filename)
    else:
      print("❗ 경고: 메타데이터 CSV 파일에 '파일명' 컬럼이 없습니다. 파일명 클리닝을 건너뜀.")
    
    print(f"메타데이터 로드 완료: {metadata_df.shape[0]}개의 공고 데이터.")
    return metadata_df
  except FileNotFoundError:
    print(f"❗ 오류: 파일을 찾을 수 없습니다: {metadata_path}.")
    return None
  except pd.errors.EmptyDataError:
    print(f"❗ 오류: 메타데이터 CSV 파일이 비어 있습니다: {metadata_path}.")
    return None
  except Exception as e:
    print(f"❗ 메타데이터 로드 중 예상치 못한 오류 발생: {e}.")
    return None

# 문서 로드 함수 수정 (사전 청크된 JSON 파일 로드용).
def load_documents_pre_chunked(file_path):
  """사전 청크된 JSON 파일에서 문서 데이터 로드"""
  
  print(f"사전 청크된 문서 데이터 로드 중: {file_path}....")
  try:
    with open(file_path, "r", encoding = "utf-8") as f:
      data = json.load(f)
    print(f"사전 청크된 데이터 로드 완료. 총 항목 수: {len(data) if isinstance(data, list) else '알 수 없음'}.")
    return data
  except FileNotFoundError:
    print(f"❗ 오류: 파일을 찾을 수 없습니다: {file_path}.")
    return None
  except json.JSONDecodeError:
    print(f"❗ 오류: JSON 디코딩 실패. 파일 형식을 확인하세요: {file_path}.")
    return None
  except Exception as e:
    print(f"❗ 데이터 로드 중 예상치 못한 오류 발생: {e}.")
    return None

def extract_numeric_amount(amount_str):
  """
  금액 문자열에서 숫자만 추출하고 단위(억, 천만, 만 등)에 따라 변환.

  Args:
    amount_str: 금액을 나타내는 문자열 (예: "1억 5천만", "10,000원", "500만").

  Returns:
    정수로 변환된 금액. 입력이 없거나 변환할 수 없으면 0 반환.
  """
  if pd.isna(amount_str) or not amount_str:
    return 0

  amount_str = str(amount_str).replace(",", "")

  # 숫자와 단위 추출.
  unit_map = {
      "억": 100000000,
      "천만": 10000000,
      "백만": 1000000,
      "만": 10000,
      "천": 1000
  }

  # 기본 숫자 추출.
  numbers = re.findall(r"\d+", amount_str)
  if not numbers:
    return 0

  # 먼저 단순 숫자만 결합.
  numeric_value = "".join(numbers)

  # 단위가 있는지 확인.
  for unit, multiplier in unit_map.items():
    if unit in amount_str:
      # 단위가 있으면 해당 단위로 변환.
      base_value = int(numeric_value)
      return base_value * multiplier

  # 단위가 없으면 그대로 반환.
  return int(numeric_value)

def preprocess_metadata(df):
  """
  메타데이터 데이터프레임 전처리.
  날짜 필드를 datetime 객체로 변환하고, 금액 필드를 숫자형으로 변환.

  Args:
    df: 전처리할 메타데이터 Pandas DataFrame.

  Returns:
    전처리된 Pandas DataFrame.
  """
  # 날짜 필드 변환.
  date_fields = ["공개 일자", "입찰 참여 시작일", "입찰 참여 마감일"]
  for field in date_fields:
    if field in df.columns:
      df[f"{field}_datetime"] = pd.to_datetime(df[field], errors = "coerce")

  # 금액 필드 변환
  if "사업 금액" in df.columns:
    df["사업 금액_숫자"] = df["사업 금액"].apply(extract_numeric_amount)

  return df

if __name__ == "__main__":
  # Add testing for extract_numeric_amount
  print("\nTesting extract_numeric_amount function....")
  test_amounts = [
    "10000",
    "10,000원",
    "500만",
    "1천만",
    "1억",
    "1억 5천만", # Test combination (might require more complex logic than current)
    "200,000,000",
    "0",
    "",
    None,
    "정보없음",
    "12345678901234567890", # Test large numbers/overflow potential
    "500만원", # Test with '원' suffix
    "천만원" # Test unit without leading digit (should parse digits only, result 0 here)
    ]

  for amount_str in test_amounts:
    numeric_amount = extract_numeric_amount(amount_str)
    print(f"Original: '{amount_str}' -> Numeric: {numeric_amount}")
  
  # Add testing for preprocess_metadata
  print("\nTesting preprocess_metadata function...")

  # Create a dummy DataFrame for testing preprocess_metadata
  dummy_preprocess_data = {
    "공고 번호": ["1", "2", "3"],
    "사업명": ["사업 A", "사업 B", "사업 C"],
    "사업 금액": ["1억", "5천만", "정보없음"],
    "발주 기관": ["기관 X", "기관 Y", "기관 Z"],
    "공개 일자": ["2023-01-01", "2023/02/15", "invalid-date"],
    "입찰 참여 시작일": ["2023-01-10", "", "2023-03-01"],
    "입찰 참여 마감일": ["2023-01-15", "2023-02-20", None],
    "파일명": ["a.pdf", "b.hwp", "c.docx"] # Include filename for context, though not processed here
  }
  dummy_preprocess_df = pd.DataFrame(dummy_preprocess_data)

  print("Original DataFrame:")
  print(dummy_preprocess_df)
  print("\nOriginal DataFrame Info:")
  dummy_preprocess_df.info()

  processed_df = preprocess_metadata(dummy_preprocess_df)

  print("\nProcessed DataFrame:")
  print(processed_df)
  print("\nProcessed DataFrame Info:")
  processed_df.info()

  # Check converted columns
  print("\nChecking converted columns:")
  print(processed_df[["공개 일자", "공개 일자_datetime", "사업 금액", "사업 금액_숫자"]])

  # Example with missing columns
  print("\nTesting preprocess_metadata with missing columns...")
  dummy_missing_cols_df = pd.DataFrame({
      "공고 번호": ["4"],
      "사업명": ["사업 D"],
      # Missing '사업 금액', '공개 일자', etc.
  })
  preprocess_metadata(dummy_missing_cols_df) # This should print warnings

  print("Testing clean_filename function....")
  test_names = [
      "example.pdf",
      "another_file.hwp",
      "document.docx ",
      "spreadsheet.XLSX",
      "데이터파일.json",
      "정규화테스트.txt", # .txt 확장자는 제거되지 않음
      "  whitespace_test.pdf  "
  ]
  for name in test_names:
    cleaned = clean_filename(name)
    print(f"Original: '{name}' → Cleaned: '{cleaned}'")

  print("Testing load_metadata function...")
  dummy_csv_content = """공고 번호,공고 차수,사업명,사업 금액,발주 기관,공개 일자,입찰 참여 시작일,입찰 참여 마감일,사업 요약,파일형식,파일명,텍스트
  202312345,00,AI 시스템 구축 사업,1000000000,과학기술정보통신부,2023-12-01,2023-12-10,2023-12-15,인공지능 시스템 구축 관련 사업입니다.,pdf,사업1.pdf,텍스트 내용 1
  202354321,01,클라우드 전환 용역,500000000,행정안전부,2023-12-05,2023-12-12,2023-12-20,기존 시스템 클라우드 전환 용역 사업입니다.,hwp,클라우드_사업.hwp,텍스트 내용 2
  202398765,00,데이터 분석 플랫폼 개발,.5억,빅데이터청,2023-12-08,2023-12-15,2023-12-22,빅데이터 분석 플랫폼 개발 사업입니다.,xlsx,데이터.xlsx ,텍스트 내용 3
  """
  dummy_metadata_path = "dummy_metadata.csv"
  with open(dummy_metadata_path, "w", encoding = "utf-8") as f:
    f.write(dummy_csv_content)

  loaded_df = load_metadata(dummy_metadata_path)

  if loaded_df is not None:
    print("\nLoaded DataFrame Head:")
    print(loaded_df.head())
    print("\nLoaded DataFrame Info:")
    loaded_df.info()

  import os
  if os.path.exists(dummy_metadata_path):
    os.remove(dummy_metadata_path)
    print(f"\nCleaned up {dummy_metadata_path}")