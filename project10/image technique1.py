import cv2
import numpy as np
import os

# 이미지가 저장된 디렉토리 경로
input_directory = r"C:\data_240422\tip_images\test\resized"

# 각 클래스에 해당하는 폴더 이름
classes = ['center', 'crush', 'cut', 'none', 'pass']

# 결과 이미지를 저장할 상위 디렉토리 경로
output_directory = r"C:\data_240422\tip_images\test\reflection_removed"

# 결과 이미지 디렉토리가 없다면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 각 클래스 폴더를 순회하며 이미지 처리
for class_name in classes:
    class_dir = os.path.join(input_directory, class_name)
    output_class_dir = os.path.join(output_directory, class_name)

    # 출력 클래스 폴더가 없다면 생성
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    # 클래스 폴더 내의 모든 이미지 파일 처리
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)

        # 이미지 읽기
        img = cv2.imread(file_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 하얀색 반사 영역을 정의하기 위한 HSV 범위 설정
        lower_white = np.array([0, 0, 168], dtype=np.uint8)
        upper_white = np.array([172, 111, 255], dtype=np.uint8)

        # HSV 이미지에서 하얀색 반사 영역에 해당하는 마스크 생성
        mask = cv2.inRange(hsv, lower_white, upper_white)

        # 마스크의 반전을 취함으로써 반사 부분을 제외하고 나머지 부분을 선택
        mask_inv = cv2.bitwise_not(mask)

        # 반사 부분을 제외한 이미지를 얻기 위해 마스크 적용
        result = cv2.bitwise_and(img, img, mask=mask_inv)

        # 결과 이미지 저장 경로
        output_file_path = os.path.join(output_class_dir, file_name)

        # 이미지 저장
        cv2.imwrite(output_file_path, result)

print("모든 클래스 폴더 내 이미지의 반사 제거 처리가 완료되었습니다.")