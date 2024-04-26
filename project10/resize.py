from PIL import Image
import os

# 원본 이미지가 있는 디렉토리 경로
directory = r"C:\data_240422\tip_images\test"

# 결과 이미지를 저장할 폴더 이름
resized_directory = 'resized'

# 결과 이미지의 크기
new_size = (244, 244)

# 디렉토리의 모든 하위 폴더를 순회
for subdir, dirs, files in os.walk(directory):
    for file in files:
        # 파일 경로 조합
        file_path = os.path.join(subdir, file)

        # 이미지 파일만 처리
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                # 이미지 크기 변경
                resized_img = img.resize(new_size, Image.ANTIALIAS)

                # 새로운 저장 폴더 경로 생성
                new_subdir = subdir.replace(directory, directory + '\\' + resized_directory)
                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir)

                # 변경된 이미지 저장
                resized_img.save(os.path.join(new_subdir, file))

print("모든 이미지의 크기가 변경되어 새로운 폴더에 저장되었습니다.")
