import cv2
import os


def apply_clahe(image_path, output_path, clip_limit=2.0, tile_grid_size=(8, 8)):
    # 이미지 로드
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 클레히 변환 객체 생성
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # 클레히 변환 적용
    clahe_img = clahe.apply(image)

    # 결과 이미지 저장
    cv2.imwrite(output_path, clahe_img)


# 디렉터리 설정
directory = r'C:\data_240422\tip_images\test\Crush'

# 디렉터리 내의 모든 이미지 파일에 대해 CLAHE 적용
for filename in os.listdir(directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 확장자 확인
        file_path = os.path.join(directory, filename)
        output_path = os.path.join(directory, 'clahe_' + filename)  # 결과 파일 이름 설정
        apply_clahe(file_path, output_path)
        print(f'Processed and saved CLAHE image for: {filename}')
