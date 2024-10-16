import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# 경로 설정
mask_dir_val = r'C:\LEE_project2024_240927_F1\val\masks'
output_image_dir = r'C:\LEE_project2024_240927_F1\val\output_images'  # 시각화된 이미지를 저장할 경로
output_excel_path = r'C:\LEE_project2024_240927_F1\val\val_results.xlsx'  # 수정된 저장 위치

# 고정된 반지름 값 설정 (300으로 고정)
fixed_circle_radius = 300

# 이미지 저장 폴더 생성
os.makedirs(output_image_dir, exist_ok=True)

# 카테고리 예측 기준 설정 (조건 중 하나라도 만족하면 해당 카테고리로 분류)
def predict_category(overall_avg_distance, min_avg_distance, max_avg_distance):
    if overall_avg_distance <= 37.57 or (0 <= min_avg_distance <= 24) or (0 <= max_avg_distance <= 50):
        return 'A'
    elif (37.57 < overall_avg_distance <= 58.51) or (25 <= min_avg_distance <= 34) or (50 <= max_avg_distance <= 90):
        return 'B'
    elif overall_avg_distance > 58.52 or min_avg_distance >= 35 or max_avg_distance >= 90:
        return 'C'
    else:
        return 'Unknown'

# 예측된 카테고리와 실제 카테고리를 비교하여 적합/부적합 판단
def check_fitting(predicted_category, true_category):
    return '적합' if predicted_category == true_category else '부적합'

# 마스크 파일을 처리하는 함수 (이미지 시각화 추가)
def process_mask(mask_file, output_image_file):
    # 마스크 이미지 로드
    mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise ValueError(f"Mask image not found: {mask_file}")

    # contours, hierarchy 사용하여 외곽선 찾기
    contours, _ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contours found in the mask image.")

    # 가장 큰 외곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 세그멘테이션된 원의 중심 찾기
    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        seg_center_x = int(moments["m10"] / moments["m00"])
        seg_center_y = int(moments["m01"] / moments["m00"])
    else:
        seg_center_x, seg_center_y = mask_image.shape[1] // 2, mask_image.shape[0] // 2

    # 기준 원의 중심 설정
    ref_circle_center = (seg_center_x, seg_center_y)

    # 외곽점에서 기준 원까지의 최소 거리를 계산하는 함수
    def calculate_distance_to_circle(point, center, radius):
        distance_to_center = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
        return abs(distance_to_center - radius)

    # 외곽선의 모든 점에서 기준 원까지의 최소 거리를 계산
    distances = []
    for contour_point in largest_contour:
        point = contour_point[0]
        min_distance = calculate_distance_to_circle(point, ref_circle_center, fixed_circle_radius)
        distances.append(min_distance)

    # 최소 거리, 최대 거리 및 전체 평균 거리 계산
    min_avg_distance = np.mean(distances)
    max_avg_distance = np.max(distances)
    overall_avg_distance = np.mean(distances)

    # 시각화된 이미지 생성 및 저장
    plt.imshow(mask_image, cmap='gray')
    plt.title(f'{mask_file} with Fixed Reference Circle')

    # 기준 원 그리기
    circle = plt.Circle(ref_circle_center, fixed_circle_radius, color='green', fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    # 외곽선 위의 점들과 기준 원까지의 거리를 표시
    for point in largest_contour:
        point = point[0]
        plt.plot(point[0], point[1], 'ro')
        plt.plot([point[0], ref_circle_center[0]], [point[1], ref_circle_center[1]], 'b-')

    plt.axis('off')

    # 시각화된 이미지를 저장
    plt.savefig(output_image_file)
    plt.close()

    return min_avg_distance, max_avg_distance, overall_avg_distance

# val 데이터 처리 함수
def process_val_data(mask_dir_val, output_excel_path, output_image_dir):
    data = []
    category_counts = {'A': {'total': 0, 'correct': 0}, 'B': {'total': 0, 'correct': 0}, 'C': {'total': 0, 'correct': 0}}

    for mask_filename in os.listdir(mask_dir_val):
        if mask_filename.endswith('.png'):  # PNG 파일만 처리
            mask_path = os.path.join(mask_dir_val, mask_filename)
            output_image_file = os.path.join(output_image_dir, mask_filename)  # 이미지 저장 경로

            try:
                # 마스크 처리 후 최소 거리, 최대 거리 및 전체 평균 거리 계산
                min_avg_distance, max_avg_distance, overall_avg_distance = process_mask(mask_path, output_image_file)

                # 파일명에서 'A', 'B', 'C'를 포함하는 실제 카테고리 추출
                if mask_filename.startswith('A'):
                    true_category = 'A'
                elif mask_filename.startswith('B'):
                    true_category = 'B'
                elif mask_filename.startswith('C'):
                    true_category = 'C'
                else:
                    true_category = 'Unknown'

                # 예측된 카테고리
                predicted_category = predict_category(overall_avg_distance, min_avg_distance, max_avg_distance)

                # 적합/부적합 판정
                fitting_result = check_fitting(predicted_category, true_category)

                # 카테고리별 적합/부적합 카운팅
                if true_category in category_counts:
                    category_counts[true_category]['total'] += 1
                    if fitting_result == '적합':
                        category_counts[true_category]['correct'] += 1

                # 결과 저장
                data.append({
                    '파일명': mask_filename,
                    '최소 거리 평균': min_avg_distance,
                    '최대 거리 평균': max_avg_distance,
                    '전체 평균 거리': overall_avg_distance,
                    '실제 카테고리': true_category,
                    '예측 카테고리': predicted_category,
                    '적합 여부': fitting_result
                })

            except Exception as e:
                print(f"Error processing {mask_path}: {e}")

    # DataFrame 생성 및 Excel 저장
    df = pd.DataFrame(data)

    # 각 카테고리별 적합 비율 계산
    accuracy_data = []
    for category, counts in category_counts.items():
        total = counts['total']
        correct = counts['correct']
        accuracy = (correct / total) * 100 if total > 0 else 0
        accuracy_data.append({
            '카테고리': category,
            '총 데이터 수': total,
            '적합 데이터 수': correct,
            '적합 비율 (%)': accuracy
        })

    # 적합 비율을 DataFrame에 추가하고 엑셀 파일로 저장
    df_accuracy = pd.DataFrame(accuracy_data)
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='결과', index=False)
        df_accuracy.to_excel(writer, sheet_name='적합 비율', index=False)

    print(f"결과가 {output_excel_path}에 저장되었습니다.")


# val 데이터 처리 실행
process_val_data(mask_dir_val, output_excel_path, output_image_dir)
