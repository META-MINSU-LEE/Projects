import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드
image = cv2.imread('C:\data\Lena.png')

# R 채널 선택 (OpenCV는 BGR 순서로 채널을 읽기 때문에 2를 사용)
R_channel = image[:, :, 2]

# R 채널 히스토그램 평탄화
R_eq = cv2.equalizeHist(R_channel)

# 이미지를 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# V 채널 히스토그램 평탄화
hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

# 평탄화된 HSV 이미지를 BGR로 다시 변환
image_eq_hsv_to_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# 결과 시각화
plt.figure(figsize=(10, 8))

# 원본 R 채널
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# 평탄화된 R 채널
plt.subplot(2, 2, 2)
plt.imshow(R_eq, cmap='gray')
plt.title('Equalized R Channel')

# HSV 변환 및 V 채널 평탄화 이미지
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(image_eq_hsv_to_bgr, cv2.COLOR_BGR2RGB))
plt.title('Image after HSV Conversion and V Equalization')

plt.tight_layout()
plt.show()