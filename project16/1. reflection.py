import cv2
import os
import numpy as np


def remove_reflection(image):
    # HSV 색상 공간으로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # V 채널 조정을 위해 CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_channel = hsv_image[:, :, 2]
    v_channel = clahe.apply(v_channel)

    # 밝기를 조금 증가시켜주는 조정 (필요에 따라 조정 가능)
    v_channel = np.clip(v_channel + 10, 0, 255)  # 밝기를 조금 증가

    # 조정된 V 채널을 다시 HSV 이미지에 적용
    hsv_image[:, :, 2] = v_channel

    # RGB로 다시 변환
    final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return final_image


def process_images(directory_path):
    # 결과 이미지를 저장할 디렉토리
    output_directory = os.path.join(directory_path, 'processed')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 디렉토리 내 모든 파일을 순회
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                processed_image = remove_reflection(image)
                # 처리된 이미지 저장
                cv2.imwrite(os.path.join(output_directory, filename), processed_image)
                print(f'Processed and saved {filename}')


# 파일 경로 설정
directory_path = r'C:\data_240422\tip_images\test\Crush'
process_images(directory_path)
