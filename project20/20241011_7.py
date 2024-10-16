import cv2
import numpy as np
import os

# 경로 설정
mask_dir = r'C:\20241013\Circle Fitting'  # 마스크 파일이 있는 경로

# 고정된 반지름 값 설정 (300으로 고정)
fixed_circle_radius = 300

# 카테고리 분류 함수 (하나라도 만족하면 분류)
def classify_category(avg_distance, min_distance, max_distance):
    # A 카테고리 조건
    if avg_distance <= 55 and min_distance <= 50 and max_distance <= 105:
        return 'A'
    # B 카테고리 조건 (더 완화된 조건)
    elif 55 < avg_distance <= 70 and 35 <= min_distance <= 70 and 80 <= max_distance <= 120:
        return 'B'
    # C 카테고리 조건 (최대 거리 강조)
    elif avg_distance > 70 and (min_distance <= 1 or min_distance >= 70) and max_distance > 120:
        return 'C'
    else:
        return 'Unclassified'

# 마스크 파일을 처리하고 거리 계산
def process_mask(mask_file):
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

    # 평균, 최소, 최대 거리 계산
    average_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    return average_distance, min_distance, max_distance

# 각 마스크 파일에 대해 카테고리 분류
def classify_masks(mask_dir):
    results = []  # 결과 저장 리스트
    correct_classifications = {'A': 0, 'B': 0, 'C': 0}  # 정확하게 분류된 파일 수
    total_classifications = {'A': 0, 'B': 0, 'C': 0}  # 실제 파일 수

    for mask_filename in os.listdir(mask_dir):
        if mask_filename.endswith('.png'):  # PNG 파일만 처리
            mask_path = os.path.join(mask_dir, mask_filename)
            try:
                avg_distance, min_distance, max_distance = process_mask(mask_path)
                category = classify_category(avg_distance, min_distance, max_distance)

                # 실제 카테고리 추정 (파일명으로 구분)
                if 'A' in mask_filename:
                    true_category = 'A'
                elif 'B' in mask_filename:
                    true_category = 'B'
                elif 'C' in mask_filename:
                    true_category = 'C'
                else:
                    true_category = 'Unclassified'

                # 총 파일 수 카운트
                if true_category in total_classifications:
                    total_classifications[true_category] += 1

                # 결과 저장
                results.append((mask_filename, category, avg_distance, min_distance, max_distance, true_category))

                # 정확한 분류 여부 카운트
                if category == true_category:
                    correct_classifications[true_category] += 1

            except Exception as e:
                print(f"Error processing {mask_path}: {e}")

    # 결과 출력 및 정확도 계산
    total_correct = sum(correct_classifications.values())
    total_files = sum(total_classifications.values())

    # 결과 출력
    for result in results:
        mask_filename, category, avg_distance, min_distance, max_distance, true_category = result
        print(f"파일: {mask_filename} -> 예측 분류: {category}, 실제 분류: {true_category}")
        print(f"  평균 거리: {avg_distance:.4f}, 최소 거리: {min_distance:.4f}, 최대 거리: {max_distance:.4f}")

    print("\n### 최종 결과 ###")
    print(f"총 파일 수: {total_files}, 정확하게 분류된 파일 수: {total_correct}")
    if total_files > 0:
        accuracy = (total_correct / total_files) * 100
        print(f"정확도: {accuracy:.2f}%")
    else:
        print("처리된 파일이 없습니다.")

    # 각 카테고리별로 맞은 개수 출력
    for category in ['A', 'B', 'C']:
        print(f"{category} 카테고리: 총 {total_classifications[category]}개 중 {correct_classifications[category]}개 정확")

# 실행
classify_masks(mask_dir)
