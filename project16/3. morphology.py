import cv2
import numpy as np
import os


def apply_morphology(image_path, output_path):
    # 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이진화 (Otsu의 자동 임계값 사용)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 모폴로지 연산을 위한 커널 생성
    kernel = np.ones((5, 5), np.uint8)

    # 열기 연산으로 작은 객체 제거 및 구멍 메우기
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 닫기 연산으로 내부 노이즈 제거
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    # 결과 이미지 저장
    cv2.imwrite(output_path, closed)


# 디렉터리 설정
directory = r'C:\data_240422\tip_images\test\Crush'

# 디렉터리 내의 모든 이미지 파일에 대해 모폴로지 연산 적용
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자 확인
        file_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, 'morphology_' + filename)  # 결과 파일 이름 설정
        apply_morphology(file_path, output_path)
        print(f'Processed and saved morphology image for: {filename}')
