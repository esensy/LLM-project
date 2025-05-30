import re
import unicodedata

def clean_filename(name: str) -> str:
    return re.sub(r'\.(pdf|hwp|docx|xlsx|json)$', '', unicodedata.normalize("NFC", name.strip()), flags=re.IGNORECASE)