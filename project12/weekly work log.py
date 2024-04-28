import pandas as pd

def merge_weekly_reports(file1, file2, week):
    try:
        data1 = pd.read_excel(file1, sheet_name=f'{week}주차', engine='openpyxl')
        data2 = pd.read_excel(file2, sheet_name=f'{week}주차', engine='openpyxl')
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 데이터 열 이름 확인
    print("Data1 Columns:", data1.columns)
    print("Data2 Columns:", data2.columns)

    # 데이터를 합칩니다.
    combined_data = pd.concat([data1, data2], ignore_index=True)

    # 열 이름 확인 및 정렬 시도
    if '이름' in combined_data.columns:
        combined_data.sort_values(by='이름', inplace=True)
    else:
        print("열 '이름'이 데이터에 존재하지 않습니다.")

    # 결과를 새로운 엑셀 파일로 저장합니다.
    output_path = file1[:file1.rfind("\\")]
    combined_data.to_excel(f"{output_path}\\합쳐진_{week}주차_업무일지.xlsx", index=False)

# 파일 경로와 주차를 지정합니다.
base_path = 'C:\\test'
file1_path = f'{base_path}\\A.xlsx'
file2_path = f'{base_path}\\B.xlsx'
week = 17

# 함수를 실행합니다.
merge_weekly_reports(file1_path, file2_path, week)