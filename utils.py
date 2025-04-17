import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def validate_excel_data(policy_df: pd.DataFrame, sales_df: pd.DataFrame) -> List[str]:
    """
    엑셀 데이터의 유효성을 검사하고 문제점 목록을 반환합니다.
    
    Args:
        policy_df: 정책 데이터프레임
        sales_df: 매출 데이터프레임
        
    Returns:
        문제점 목록
    """
    issues = []
    
    # 1. 업체명 일치 여부 확인
    policy_companies = set(policy_df['업체명'].unique())
    sales_companies = set(sales_df['업체명'].unique())
    
    missing_in_policy = sales_companies - policy_companies
    if missing_in_policy:
        issues.append(f"다음 업체들이 정책 시트에 없습니다: {', '.join(missing_in_policy)}")
    
    # 2. 수수료율 데이터 타입 확인
    if not pd.api.types.is_numeric_dtype(policy_df['수수료율']):
        issues.append("수수료율 열이 숫자 형식이 아닙니다.")
    elif (policy_df['수수료율'] < 0).any() or (policy_df['수수료율'] > 100).any():
        issues.append("일부 수수료율이 유효하지 않습니다 (0-100% 범위를 벗어남)")
    
    # 3. 매출액 데이터 타입 확인
    if not pd.api.types.is_numeric_dtype(sales_df['매출액']):
        issues.append("매출액 열이 숫자 형식이 아닙니다.")
    elif (sales_df['매출액'] < 0).any():
        issues.append("일부 매출액이 음수입니다.")
    
    return issues

def format_currency(amount: float) -> str:
    """
    금액을 통화 형식으로 포맷팅합니다.
    
    Args:
        amount: 금액
        
    Returns:
        포맷팅된 금액 문자열
    """
    return f"{amount:,.0f}원"

def extract_numeric_value(text: str) -> float:
    """
    텍스트에서 숫자 값을 추출합니다.
    
    Args:
        text: 숫자가 포함된 텍스트
        
    Returns:
        추출된 숫자 값
    """
    import re
    
    # 숫자와 소수점만 추출
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    if numbers:
        return float(numbers[0])
    return 0.0
