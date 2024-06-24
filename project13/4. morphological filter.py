import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드
image = cv2.imread('C:\data\Lena.png', cv2.IMREAD_GRAYSCALE)

# 사용자 입력 가정
# 'otsu' 또는 'adaptive' 중 선택
binarization_method = 'otsu'
# 'erosion', 'dilation', 'opening', 'closing' 중 선택
morphology_operation = 'opening'
# 적용 횟수
num_iterations = 2

# Otsu's 이진화
if binarization_method == 'otsu':
    _, bin_img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Adaptive 이진화
elif binarization_method == 'adaptive':
    bin_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

# 모폴로지 연산
kernel = np.ones((5, 5), np.uint8)
if morphology_operation == 'erosion':
    result_img = cv2.erode(bin_img, kernel, iterations=num_iterations)
elif morphology_operation == 'dilation':
    result_img = cv2.dilate(bin_img, kernel, iterations=num_iterations)
elif morphology_operation == 'opening':
    result_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=num_iterations)
elif morphology_operation == 'closing':
    result_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=num_iterations)

# 결과 시각화
plt.figure(figsize=(10, 7))

plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(bin_img, cmap='gray'), plt.title('Binarized Image')
plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(result_img, cmap='gray'), plt.title('Morphology Result')
plt.xticks([]), plt.yticks([])

plt.show()