import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드 및 그레이스케일 변환
image = cv2.imread('C:\data\Lena.png', cv2.IMREAD_GRAYSCALE)

# DFT를 수행하고 주파수 변환
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# 주파수 도메인 이미지의 크기
rows, cols = image.shape
crow, ccol = rows // 2 , cols // 2

# 사용자로부터 반지름 입력 받기
# 예제에서는 가상의 값으로 대체합니다. 실제 구현시 사용자 입력을 받아야 합니다.
r1, r2 = 30, 60  # 예제 반지름 값

# 마스크 생성, 두 원 사이의 영역에 1을 설정
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), r2, (1, 1), thickness=-1)
cv2.circle(mask, (ccol, crow), r1, (0, 0), thickness=-1)

# 마스크 적용
fshift = dft_shift * mask

# 역 DFT로 이미지 복원
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# 결과 시각화
plt.figure(figsize=(12, 6))

plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])), cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(img_back, cmap='gray')
plt.title('Band Pass Filtered Image'), plt.xticks([]), plt.yticks([])

plt.show()