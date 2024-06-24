import cv2
import os
import numpy as np
from scipy.io import loadmat

videosFrameFolder = 'output/images'
labelsDataFolder = 'output/labels'

w = 1920
h = 1080


def print_folder_name_part(folder_path):
    folder_name = os.path.basename(folder_path)
    parts = folder_name.split('_')

    # Join the first three parts or all parts if less than three
    result = '_'.join(parts[:min(3, len(parts))])
    return result


def video_to_frames(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    success, image = cap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(output_path, f"frame_{count:05d}.jpg"), image)  # save frame as JPEG file
        success, image = cap.read()
        count += 1
        print('Read a new frame: ', output_path)


def matlab_to_txt(input_path, output_path):
    data = loadmat(input_path)
    for i in range(data['structXML'].shape[1]):
        filename = os.path.join(output_path, f'frame_{i:05d}.txt')
        with open(filename, 'w') as f:
            if data['structXML'][0, i]['BB'].size > 0:
                for j in range(data['structXML'][0, i]['BB'].shape[0]):
                    x_center = (data['structXML'][0, i]['BB'][j, 0] + (data['structXML'][0, i]['BB'][j, 2] / 2)) / w
                    y_center = (data['structXML'][0, i]['BB'][j, 1] + (data['structXML'][0, i]['BB'][j, 3] / 2)) / h
                    width = data['structXML'][0, i]['BB'][j, 2] / w
                    height = data['structXML'][0, i]['BB'][j, 3] / h
                    class_index = int(data['structXML'][0, i]['Object'][j]) - 1
                    f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")


folders = ['VIS_Onboard', 'VIS_Onshore']

for folder in folders:
    video_folder = os.path.join('smd_plus', folder, 'Videos')
    for video_file in os.listdir(video_folder):
        video_path = os.path.join(video_folder, video_file)
        video_frame_output_path = os.path.join(videosFrameFolder,
                                               print_folder_name_part(os.path.splitext(video_file)[0]))
        os.makedirs(video_frame_output_path, exist_ok=True)
        video_to_frames(video_path, video_frame_output_path)

    matlab_folder = os.path.join('smd_plus', folder, 'ObjectGT')
    for matlab_file in os.listdir(matlab_folder):
        matlab_path = os.path.join(matlab_folder, matlab_file)
        label_data_output_path = os.path.join(labelsDataFolder,
                                              print_folder_name_part(os.path.splitext(matlab_file)[0]))
        os.makedirs(label_data_output_path, exist_ok=True)
        matlab_to_txt(matlab_path, label_data_output_path)
