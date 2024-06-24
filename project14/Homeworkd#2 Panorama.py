import cv2
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
image_sets = [
    ['stitching/boat1.jpg', 'stitching/boat2.jpg', 'stitching/boat3.jpg', 'stitching/boat4.jpg'],
    ['stitching/budapest1.jpg', 'stitching/budapest2.jpg', 'stitching/budapest3.jpg', 'stitching/budapest4.jpg'],
    ['stitching/newspaper1.jpg', 'stitching/newspaper2.jpg', 'stitching/newspaper3.jpg', 'stitching/newspaper4.jpg'],
    ['stitching/s1.jpg', 'stitching/s2.jpg']
]


# Stitcher 객체 생성 함수
def create_stitcher():
    try:
        return cv2.Stitcher.create()
    except AttributeError:
        return cv2.createStitcher()


# Stitcher 객체 생성
stitcher = create_stitcher()


# 파노라마 이미지 생성 함수
def create_panorama(image_paths):
    images = [cv2.imread(os.path.join(extract_path, image_path)) for image_path in image_paths]

    if any(img is None for img in images):
        print("Error: One or more images are not loaded correctly.")
        return

    # 파노라마 생성
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
        plt.title('Panorama Image')
        plt.axis('off')
        plt.show()
    else:
        print("Error during stitching: ", status)


# 각 이미지 셋에 대해 파노라마 이미지 생성
for image_set in image_sets:
    create_panorama(image_set)
