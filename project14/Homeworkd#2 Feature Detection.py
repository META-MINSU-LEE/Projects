import cv2
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import os

# 압축 파일 경로와 압축 해제 경로 설정
zip_file_path = r'C:\data240609\Stitching.zip'
extract_path = r'C:\data240609\Stitching'

# 압축 파일 해제
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# 이미지를 로드할 파일 경로
image_files = ['stitching/boat1.jpg', 'stitching/budapest1.jpg', 'stitching/newspaper1.jpg', 'stitching/s1.jpg']
image_paths = [os.path.join(extract_path, file) for file in image_files]

# Canny Edge Detection 및 Harris Corner Detection 수행 및 결과 출력 함수
def detect_features(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image {image_path}")
        return

    # Canny Edge Detection
    edges = cv2.Canny(image, 100, 200)

    # Harris Corner Detection
    dst = cv2.cornerHarris(image, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image_harris = np.copy(image)
    image_harris[dst > 0.01 * dst.max()] = 255

    # 결과 출력
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image_harris, cmap='gray')
    plt.title('Harris Corner Detection')
    plt.axis('off')

    plt.show()

# 각 이미지에 대해 특징 검출 및 결과 출력
for image_path in image_paths:
    detect_features(image_path)


