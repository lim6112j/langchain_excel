import pandas as pd
import numpy as np
import os

# 샘플 데이터 생성 함수
def create_sample_excel():
    # 정책 데이터 생성
    policy_data = {
        '업체명': ['A상사', 'B마트', 'C스토어', 'D컴퍼니', 'E마켓'],
        '수수료율': [10, 15, 12, 8, 20],
        '정산주기': ['월간', '주간', '월간', '월간', '주간'],
        '프로모션': ['Y', 'N', 'Y', 'N', 'Y'],
        '특별할인': ['N', 'Y', 'N', '확인필요: 할인율 미정', 'N'],
        '부가서비스': ['배송지원', '없음', '확인필요: 서비스 종류 확인 필요', '마케팅지원', '없음']
    }
    
    # 매출 데이터 생성
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    sales_data = []
    for company in policy_data['업체명']:
        # 각 업체별로 5-10개의 매출 데이터 생성
        num_entries = np.random.randint(5, 11)
        for _ in range(num_entries):
            sales_amount = np.random.randint(100000, 5000000)
            date = f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            product = f"상품{np.random.randint(1, 100):03d}"
            
            sales_data.append({
                '업체명': company,
                '날짜': date,
                '상품명': product,
                '매출액': sales_amount
            })
    
    # DataFrame 생성
    policy_df = pd.DataFrame(policy_data)
    sales_df = pd.DataFrame(sales_data)
    
    # 엑셀 파일로 저장
    with pd.ExcelWriter('sample_data.xlsx') as writer:
        policy_df.to_excel(writer, sheet_name='정책', index=False)
        sales_df.to_excel(writer, sheet_name='매출', index=False)
    
    print("샘플 데이터가 'sample_data.xlsx' 파일로 생성되었습니다.")

if __name__ == "__main__":
    create_sample_excel()
