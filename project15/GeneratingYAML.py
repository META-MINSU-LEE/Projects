import os
import shutil
import yaml


def collect_data_from_directories(base_images_dir, base_labels_dir, subdirs):
    data = []

    for subdir in subdirs:
        video_images_dir = os.path.join(base_images_dir, subdir)
        video_labels_dir = os.path.join(base_labels_dir, subdir)

        if not os.path.isdir(video_images_dir) or not os.path.isdir(video_labels_dir):
            continue

        for image_file in os.listdir(video_images_dir):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(video_images_dir, image_file)
                label_path = os.path.join(video_labels_dir, image_file.replace('frame_', 'frame_').replace('.jpg', '.txt'))

                if os.path.exists(label_path):
                    data.append((image_path, label_path))

    return data


def create_yolo_dataset(base_dir, train_subdirs, val_subdirs, output_yaml_file):
    # Get the parent directory of the datasets folder
    parent_dir = os.path.dirname(os.path.dirname(output_yaml_file))

    # Create directories if they don't exist
    os.makedirs(os.path.join(parent_dir, 'datasets', 'smd_plus', 'train'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'datasets', 'smd_plus', 'val'), exist_ok=True)

    # Collect and copy training data
    train_data = collect_data_from_directories(os.path.join(base_dir, 'images'), os.path.join(base_dir, 'labels'), train_subdirs)
    for image_path, label_path in train_data:
        image_filename = os.path.basename(image_path)
        label_filename = os.path.basename(label_path)
        source_dir = os.path.basename(os.path.dirname(image_path))

        new_image_path = os.path.join(parent_dir, 'datasets', 'smd_plus', 'train', f'{source_dir}_{image_filename}')
        new_label_path = os.path.join(parent_dir, 'datasets', 'smd_plus', 'train', f'{source_dir}_{label_filename}')

        shutil.copy(image_path, new_image_path)
        shutil.copy(label_path, new_label_path)

    # Collect and copy validation data
    val_data = collect_data_from_directories(os.path.join(base_dir, 'images'), os.path.join(base_dir, 'labels'), val_subdirs)
    for image_path, label_path in val_data:
        image_filename = os.path.basename(image_path)
        label_filename = os.path.basename(label_path)
        source_dir = os.path.basename(os.path.dirname(image_path))

        new_image_path = os.path.join(parent_dir, 'datasets', 'smd_plus', 'val', f'{source_dir}_{image_filename}')
        new_label_path = os.path.join(parent_dir, 'datasets', 'smd_plus', 'val', f'{source_dir}_{label_filename}')

        shutil.copy(image_path, new_image_path)
        shutil.copy(label_path, new_label_path)

    # Create the YAML structure
    data_yaml = {
        'path': '../datasets/smd_plus',
        'train': 'train',
        'val': 'val',
        'nc': 7,  # Number of classes
        'names': ['Ferry', 'Buoy', 'Vessel_ship', 'Boat', 'Kayak', 'Sail_boat', 'Other']  # Class names
    }

    # Save to YAML file
    with open(output_yaml_file, 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=False)

    print(f"YAML file generated and saved to {output_yaml_file}")


if __name__ == "__main__":
    base_directory = "output"

    train_subdirectories = ['MVI_1451_VIS', 'MVI_1452_VIS', 'MVI_1470_VIS', 'MVI_1471_VIS', 'MVI_1478_VIS',
                            'MVI_1479_VIS', 'MVI_1481_VIS', 'MVI_1482_VIS', 'MVI_1483_VIS', 'MVI_1484_VIS',
                            'MVI_1485_VIS', 'MVI_1486_VIS', 'MVI_1578_VIS', 'MVI_1582_VIS', 'MVI_1583_VIS',
                            'MVI_1584_VIS', 'MVI_1609_VIS', 'MVI_1610_VIS', 'MVI_1619_VIS', 'MVI_1612_VIS',
                            'MVI_1617_VIS', 'MVI_1620_VIS', 'MVI_1622_VIS', 'MVI_1623_VIS', 'MVI_1624_VIS',
                            'MVI_1625_VIS', 'MVI_1626_VIS', 'MVI_1627_VIS', 'MVI_0788_VIS', 'MVI_0789_VIS',
                            'MVI_0790_VIS', 'MVI_0792_VIS', 'MVI_0794_VIS', 'MVI_0795_VIS', 'MVI_0796_VIS',
                            'MVI_0797_VIS', 'MVI_0801_VIS']
    val_subdirectories = ['MVI_1469_VIS', 'MVI_1474_VIS', 'MVI_1587_VIS', 'MVI_1592_VIS', 'MVI_1613_VIS',
                          'MVI_1614_VIS', 'MVI_1615_VIS', 'MVI_1644_VIS', 'MVI_1645_VIS', 'MVI_1646_VIS',
                          'MVI_1448_VIS', 'MVI_1640_VIS', 'MVI_0799_VIS', 'MVI_0804_VIS']

    output_yaml_file = "smd_plus.yaml"

    create_yolo_dataset(base_directory, train_subdirectories, val_subdirectories, output_yaml_file)
