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
image_pairs = [
    ('stitching/boat1.jpg', 'stitching/boat2.jpg'),
    ('stitching/budapest1.jpg', 'stitching/budapest2.jpg'),
    ('stitching/newspaper1.jpg', 'stitching/newspaper2.jpg'),
    ('stitching/s1.jpg', 'stitching/s2.jpg')
]

# 특징점 추출 및 매칭 함수
def feature_matching(image_path1, image_path2, method='ORB'):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if method == 'SIFT':
        detector = cv2.SIFT_create()
    elif method == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()
    elif method == 'ORB':
        detector = cv2.ORB_create()
    else:
        raise ValueError("Unknown method: {}".format(method))

    # 특징점 및 디스크립터 추출
    kp1, des1 = detector.detectAndCompute(image1, None)
    kp2, des2 = detector.detectAndCompute(image2, None)

    # BFMatcher를 사용한 매칭
    if method in ['SIFT', 'SURF']:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    else:  # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭 결과 이미지 출력
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.imshow(img_matches)
    plt.title(f'{method} Feature Matching')
    plt.axis('off')
    plt.show()

    # 좋은 매칭점 선택
    good_matches = matches[:50]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 호모그래피 계산 및 RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = image1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)

    # 워핑된 이미지 출력
    img_warped = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
    plt.figure(figsize=(15, 10))
    plt.subplot(121)
    plt.imshow(image1, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(img_warped, cmap='gray')
    plt.title('Warped Image')
    plt.axis('off')
    plt.show()

# 각 이미지 쌍에 대해 특징점 추출 및 매칭 수행
for image_path1, image_path2 in image_pairs:
    for method in ['ORB']:  # SIFT와 SURF는 사용하지 않음
        feature_matching(os.path.join(extract_path, image_path1), os.path.join(extract_path, image_path2), method)


