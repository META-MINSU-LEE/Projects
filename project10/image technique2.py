import cv2
import numpy as np
import os

# 데이터 경로 설정
data_path = r"C:\data_240422\tip_images\test\resized"

# 결과 이미지를 저장할 상위 디렉토리 경로
output_directory = r"C:\data_240422\tip_images\test\processed_images"

# 결과 이미지 디렉토리가 없다면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 각 클래스 폴더를 순회하며 이미지 처리
for class_name in ['center', 'crush', 'cut', 'none', 'pass']:
    class_folder = os.path.join(data_path, class_name)
    class_output_folder = os.path.join(output_directory, class_name)

    # 출력 클래스 폴더가 없다면 생성
    if not os.path.exists(class_output_folder):
        os.makedirs(class_output_folder)

    # 클래스 폴더 내의 모든 이미지 파일 처리
    for file_name in os.listdir(class_folder):
        file_path = os.path.join(class_folder, file_name)
        img = cv2.imread(file_path)

        # 가우시안 블러 적용
        gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Canny 엣지 검출 적용
        edges = cv2.Canny(img, 100, 200)

        # 적응적 이진화 적용
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 모폴로지 연산 적용 (침식)
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(gray, kernel, iterations=1)

        # 모폴로지 연산 적용 (팽창)
        dilation = cv2.dilate(gray, kernel, iterations=1)

        # Histogram Equalization 적용
        hist_eq = cv2.equalizeHist(gray)

        # 결과 이미지 저장
        cv2.imwrite(os.path.join(class_output_folder, f"{file_name}_gaussian.jpg"), gaussian_blur)
        cv2.imwrite(os.path.join(class_output_folder, f"{file_name}_edges.jpg"), edges)
        cv2.imwrite(os.path.join(class_output_folder, f"{file_name}_adaptive_thresh.jpg"), adaptive_thresh)
        cv2.imwrite(os.path.join(class_output_folder, f"{file_name}_erosion.jpg"), erosion)
        cv2.imwrite(os.path.join(class_output_folder, f"{file_name}_dilation.jpg"), dilation)
        cv2.imwrite(os.path.join(class_output_folder, f"{file_name}_hist_eq.jpg"), hist_eq)

print("모든 이미지 처리가 완료되었습니다.")
