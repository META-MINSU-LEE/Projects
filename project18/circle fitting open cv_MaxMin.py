import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 경로 설정
mask_dir_train = r'C:\LEE_project2024_240927_F1\train\masks'
output_dir_train = r'C:\LEE_project2024_240927_F1\train\output'

# 디렉토리 생성
os.makedirs(output_dir_train, exist_ok=True)

# 고정된 반지름 값 설정 (300으로 고정)
fixed_circle_radius = 300

# 마스크 파일을 처리하는 함수
def process_mask(mask_file, output_dir):
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

    # 계산된 거리를 시각화 (출력 폴더에 저장)
    plt.imshow(mask_image, cmap='gray')
    plt.title(f'{mask_file} with Fixed Reference Circle')

    # 기준 원 그리기 (고정된 반지름 사용)
    circle = plt.Circle(ref_circle_center, fixed_circle_radius, color='green', fill=False, linewidth=2)
    plt.gca().add_artist(circle)

    # 외곽선 위의 점들과 기준 원까지의 거리를 표시
    for point in largest_contour:
        point = point[0]
        plt.plot(point[0], point[1], 'ro')
        plt.plot([point[0], ref_circle_center[0]], [point[1], ref_circle_center[1]], 'b-')

    plt.axis('off')

    # 파일 저장
    output_path = os.path.join(output_dir, os.path.basename(mask_file))
    plt.savefig(output_path)
    plt.close()

    # 평균, 최소, 최대 거리 계산
    average_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    return average_distance, min_distance, max_distance


# 각 클래스(A, B, C)별로 파일을 처리
classes = {'A': [], 'B': [], 'C': []}

for mask_filename in os.listdir(mask_dir_train):
    if mask_filename.endswith('.png'):  # PNG 파일만 처리
        if 'A' in mask_filename:
            classes['A'].append(mask_filename)
        elif 'B' in mask_filename:
            classes['B'].append(mask_filename)
        elif 'C' in mask_filename:
            classes['C'].append(mask_filename)

# 클래스별 평균 거리, 최소 거리, 최대 거리 계산
for cls in classes:
    avg_distances = []
    min_distances = []
    max_distances = []

    for mask_filename in classes[cls]:
        mask_path = os.path.join(mask_dir_train, mask_filename)
        try:
            avg_distance, min_distance, max_distance = process_mask(mask_path, output_dir_train)
            avg_distances.append(avg_distance)
            min_distances.append(min_distance)
            max_distances.append(max_distance)
        except Exception as e:
            print(f"Error processing {mask_path}: {e}")

    if avg_distances:
        overall_avg_distance = np.mean(avg_distances)
        overall_min_distance = np.min(min_distances)
        overall_max_min_distance = np.max(min_distances)  # 최소 거리 중 최대값
        overall_min_max_distance = np.min(max_distances)  # 최대 거리 중 최소값
        overall_max_distance = np.max(max_distances)

        print(f"{cls} 카테고리 평균 거리: {overall_avg_distance}")
        print(f"{cls} 카테고리 최소 거리 중 최소: {overall_min_distance}")
        print(f"{cls} 카테고리 최소 거리 중 최대: {overall_max_min_distance}")
        print(f"{cls} 카테고리 최대 거리 중 최소: {overall_min_max_distance}")
        print(f"{cls} 카테고리 최대 거리 중 최대: {overall_max_distance}")
    else:
        print(f"{cls} 클래스의 유효한 데이터가 없습니다.")
