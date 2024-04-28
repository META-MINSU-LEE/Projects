import pandas as pd
from datetime import datetime

# 초기 데이터 설정
data = {
    '이름': ['이민수', '박종태', '강병훈'],
    '초기 휴가 개수': [20, 25, 27]
}

# DataFrame 생성
df = pd.DataFrame(data)
df['잔여 휴가'] = df['초기 휴가 개수']  # 초기에는 모든 휴가가 남아있음

# 휴가 사용 기록을 저장할 DataFrame 생성
df_history = pd.DataFrame(columns=['이름', '날짜', '초기 휴가 개수', '사용 휴가', '잔여 휴가'])

def use_vacation(name, date, vacation_used):
    global df, df_history
    # 휴가 사용자 검색
    mask = df['이름'] == name
    # 잔여 휴가 차감
    df.loc[mask, '잔여 휴가'] -= vacation_used
    # 휴가 기록 추가
    initial_vacations = df.loc[mask, '초기 휴가 개수'].values[0]
    remaining_vacation = df.loc[mask, '잔여 휴가'].values[0]
    new_record = {
        '이름': name,
        '날짜': date,
        '초기 휴가 개수': initial_vacations,
        '사용 휴가': vacation_used,
        '잔여 휴가': remaining_vacation
    }
    df_history = df_history.append(new_record, ignore_index=True)

# 휴가 사용 예시
use_vacation('이민수', '2024-01-02', 1)
use_vacation('이민수', '2024-01-25', 0.5)
use_vacation('이민수', '2024-02-15', 1)
use_vacation('이민수', '2024-03-2', 1)
use_vacation('박종태', '2024-01-22', 0.5)
use_vacation('박종태', '2024-03-04', 0.5)
use_vacation('박종태', '2024-03-05', 1)
use_vacation('강병훈', '2024-04-22', 1)

# 결과 출력
print(df_history)

# 엑셀 파일로 저장
df_history.to_excel('vacation_records.xlsx', index=False)