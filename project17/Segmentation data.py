import os
import cv2
import json
import numpy as np

# 경로 설정
train_image_dir = 'C:/20241011/train_jpg'  # 원본 이미지 경로
train_json_dir = 'C:/20241011/train_json'  # 원본 JSON 파일 경로
save_image_dir = 'C:/20241011/augmented_train_jpg'  # 증강된 이미지 저장 경로
save_mask_dir = 'C:/20241011/augmented_train_mask'  # 증강된 마스크 파일 저장 경로

# 폴더가 없으면 생성
os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)

# 회전 각도 설정
angles = range(30, 360, 30)

# 라벨명 설정 함수 (파일명에 따른 라벨 지정)
def get_label_from_filename(filename):
    if filename.startswith('A'):
        return 'Acircle'
    elif filename.startswith('B'):
        return 'Bcircle'
    elif filename.startswith('C'):
        return 'Ccircle'
    else:
        return None

# 이미지 및 폴리곤을 회전시키고 마스크 생성
def rotate_image_and_create_mask(image_path, json_path, angle, save_image_dir, save_mask_dir):
    # 이미지 파일 및 JSON 파일 이름 설정
    image_filename = os.path.basename(image_path)
    json_filename = os.path.basename(json_path)

    # 이미지 불러오기
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 이미지 중심 좌표
    center = (w // 2, h // 2)

    # 이미지 회전 행렬 생성
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 이미지 회전
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # JSON 파일 읽기
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 빈 마스크 생성 (0으로 채워진 흑백 이미지)
    mask = np.zeros((h, w), dtype=np.uint8)

    # polygon 좌표 회전 후 마스크에 그리기
    label = get_label_from_filename(image_filename)  # 파일명에 따른 라벨 설정
    if label:
        for shape in data['shapes']:
            if shape['label'] == label:
                points = np.array(shape['points'], dtype=np.float32)
                # 회전된 좌표 계산
                for i in range(len(points)):
                    point = np.dot(M, np.array([points[i][0], points[i][1], 1]))
                    points[i] = point[:2]
                # 회전된 폴리곤을 마스크에 그리기
                points = points.astype(np.int32)
                cv2.fillPoly(mask, [points], 255)

    # 회전된 이미지 및 마스크 저장
    rotated_image_filename = image_filename.replace('.jpg', f'_rotated_{angle}.jpg')
    rotated_mask_filename = image_filename.replace('.jpg', f'_rotated_{angle}_mask.png')

    cv2.imwrite(os.path.join(save_image_dir, rotated_image_filename), rotated_image)
    cv2.imwrite(os.path.join(save_mask_dir, rotated_mask_filename), mask)

# 모든 이미지와 JSON 파일에 대해 회전 및 마스크 생성 수행
for image_filename in os.listdir(train_image_dir):
    if image_filename.endswith('.jpg'):
        image_path = os.path.join(train_image_dir, image_filename)
        json_path = os.path.join(train_json_dir, image_filename.replace('.jpg', '.json'))

        # 각도를 변경하면서 이미지 및 마스크 증강
        for angle in angles:
            rotate_image_and_create_mask(image_path, json_path, angle, save_image_dir, save_mask_dir)
