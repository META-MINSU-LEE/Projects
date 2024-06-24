import cv2
import os


def binarize_image(image_path, output_path):
    # 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화 (Otsu의 자동 임계값 사용)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 결과 이미지 저장
    cv2.imwrite(output_path, binary)


# 디렉터리 설정
directory = r'C:\data_240422\tip_images\test\Crush'

# 디렉터리 내의 모든 이미지 파일 처리
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자 확인
        file_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, 'binary_' + filename)  # 저장할 파일 이름 설정
        binarize_image(file_path, output_path)
        print(f'Processed and saved binary image for: {filename}')