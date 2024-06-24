import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드
image = cv2.imread('C:\data\Lena.png')

# 노이즈 추가 함수
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_image = image + gauss * 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# 노이즈 추가
noisy_image = add_gaussian_noise(image)

# 필터링 적용
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)
median_filtered = cv2.medianBlur(noisy_image, 5)
bilateral_filtered = cv2.bilateralFilter(noisy_image, 9, 75, 75)

# 절대값 차이 계산
abs_diff_gaussian = cv2.absdiff(image, gaussian_filtered)
abs_diff_median = cv2.absdiff(image, median_filtered)
abs_diff_bilateral = cv2.absdiff(image, bilateral_filtered)

# 결과 시각화
plt.figure(figsize=(15, 10))

titles = ['Original Image', 'Noisy Image', 'Gaussian Filtered', 'Median Filtered', 'Bilateral Filtered',
          'Abs Diff Gaussian', 'Abs Diff Median', 'Abs Diff Bilateral']
images = [image, noisy_image, gaussian_filtered, median_filtered, bilateral_filtered,
          abs_diff_gaussian, abs_diff_median, abs_diff_bilateral]

for i in range(8):
    plt.subplot(3, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()