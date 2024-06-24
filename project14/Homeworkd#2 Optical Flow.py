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

# 이미지 파일 경로 설정
image_paths = ['stitching/dog_a.jpg', 'stitching/dog_b.jpg']
image1_path = os.path.join(extract_path, image_paths[0])
image2_path = os.path.join(extract_path, image_paths[1])

# 이미지 읽기
image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Good Features to Track 검출
features_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
p0 = cv2.goodFeaturesToTrack(image1, mask=None, **features_params)

# Pyramid Lucas-Kanade Optical Flow 계산
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

p1, st, err = cv2.calcOpticalFlowPyrLK(image1, image2, p0, None, **lk_params)

# Optical Flow 결과를 이미지에 그리기
def draw_optical_flow(img, p0, p1, st):
    mask = np.zeros_like(img)
    for i, (new, old) in enumerate(zip(p1[st == 1], p0[st == 1])):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color=(0, 255, 0), thickness=2)
        img = cv2.circle(img, (a, b), 5, color=(0, 255, 0), thickness=-1)
    return cv2.add(img, mask)

image1_color = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
optical_flow_image = draw_optical_flow(image1_color, p0, p1, st)

# Optical Flow 결과 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(optical_flow_image, cv2.COLOR_BGR2RGB))
plt.title('Optical Flow using Pyramid Lucas-Kanade')
plt.axis('off')
plt.show()
