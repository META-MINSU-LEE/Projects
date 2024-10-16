import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt

# 경로 설정
train_image_dir = 'C:/20241013/train_jpg'
train_mask_dir = 'C:/20241013/train_mask'
augmented_train_image_dir = 'C:/20241013/augmented_train_jpg'
augmented_train_mask_dir = 'C:/20241013/augmented_train_mask'
val_image_dir = 'C:/20241013/val_jpg'
val_mask_dir = 'C:/20241013/val_mask'
model_save_path = 'C:/20241013/deeplabv3_model.pth'


# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_filenames = os.listdir(image_dir)
        self.mask_filenames = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 마스크는 흑백(L)으로 변환

        if self.transform:
            image = self.transform(image)  # 이미지 변환 (예: 512x512로 리사이즈)

        if self.mask_transform:
            mask = self.mask_transform(mask)  # 마스크도 동일하게 변환 (512x512로 리사이즈)

        return image, mask


# 데이터 전처리 정의 (이미지 및 마스크 크기를 512x512로 수정)
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 이미지 크기를 512x512로 줄임
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 마스크 크기를 512x512로 줄임
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 이미지 크기를 512x512로 줄임
    transforms.ToTensor()
])

# 학습용 데이터셋: 원본 + 증강된 데이터 합치기
train_dataset = CustomDataset(train_image_dir, train_mask_dir, transform=train_transform, mask_transform=mask_transform)
augmented_train_dataset = CustomDataset(augmented_train_image_dir, augmented_train_mask_dir, transform=train_transform, mask_transform=mask_transform)
combined_train_dataset = torch.utils.data.ConcatDataset([train_dataset, augmented_train_dataset])

# 검증용 데이터셋
val_dataset = CustomDataset(val_image_dir, val_mask_dir, transform=val_transform, mask_transform=mask_transform)

# DataLoader 정의 (batch_size를 2로 줄이고, num_workers 설정)
train_loader = DataLoader(combined_train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

# 모델 정의 (pre-trained DeepLabv3 with ResNet-50 backbone)
weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1  # 사용할 weight 설정
model = deeplabv3_resnet50(weights=weights)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # 출력 채널 수 2개로 변경 (원형 클래스 + 배경)
model = model.cuda()  # GPU 사용

# 손실 함수 및 최적화 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# 학습 및 검증 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    total_start_time = time.time()  # 전체 학습 시작 시간

    for epoch in range(num_epochs):
        start_time = time.time()  # 에포크 시작 시간 측정

        # 모델 학습 모드
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, masks in train_loader:
            images = images.cuda()
            masks = masks.long().squeeze(1).cuda()  # 마스크의 첫 번째 차원 제거

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy 계산
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == masks).sum().item()
            total_train += masks.numel()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # 모델 검증 모드
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.cuda()
                masks = masks.long().squeeze(1).cuda()  # 마스크의 첫 번째 차원 제거

                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Accuracy 계산
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == masks).sum().item()
                total_val += masks.numel()

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = correct_val / total_val
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        # 에포크 시간 측정
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}, "
              f"Time: {epoch_time:.2f}s")

    # 전체 학습 시간 측정
    total_time = time.time() - total_start_time
    print(f"전체 학습 시간: {total_time / 60:.2f} 분")  # 시간을 분 단위로 출력

    # 모델 저장
    torch.save(model.state_dict(), model_save_path)
    print(f"모델이 {model_save_path}에 저장되었습니다.")

    # 학습 및 검증 결과 시각화
    plt.figure(figsize=(12, 6))

    # Loss 시각화
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    # Accuracy 시각화
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.show()


# 이 부분 추가: main 블록 안에서 실행
if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

