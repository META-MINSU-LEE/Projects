import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50

# 경로 설정
test_image_dir = 'C:/20241013/test_jpg'  # 테스트 이미지 경로
test_mask_dir = 'C:/20241013/test_mask'  # 실제 마스크 경로
model_save_path = 'C:/20241013/deeplabv3_model.pth'  # 학습된 모델 저장 경로
prediction_save_dir = 'C:/20241013/predicted_masks'  # 예측된 마스크 저장 경로

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
        self.mask_filenames = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # 원본 이미지 크기 가져오기
        image_original = Image.open(img_path).convert("RGB")
        mask_original = Image.open(mask_path).convert("L")

        image = image_original
        mask = mask_original

        # 이미지를 512x512로 리사이즈하여 모델 입력으로 사용
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)

        return image, mask, image_original.size, os.path.basename(img_path)  # 원본 이미지 크기 및 파일명 반환

# 데이터 전처리 정의 (512x512로 리사이즈)
test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 테스트용 데이터셋 및 DataLoader 정의
test_dataset = CustomDataset(test_image_dir, test_mask_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 모델 로드
model = deeplabv3_resnet50(weights=None)  # pre-trained 없이 초기화
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1))  # 두 클래스 (원형 + 배경)
model.load_state_dict(torch.load(model_save_path), strict=False)  # 학습된 모델 가중치 로드
model = model.cuda()  # GPU 사용
model.eval()  # 평가 모드

# IOU 계산 함수
def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask)
    union = np.logical_or(pred_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# 예측된 마스크 저장 함수
def save_prediction_mask(pred_mask_np, original_size, filename, save_dir):
    prediction_resized = Image.fromarray(pred_mask_np.astype(np.uint8) * 255)
    prediction_resized = prediction_resized.resize(original_size, Image.NEAREST)  # 원본 크기
    save_path = os.path.join(save_dir, f"pred_{filename}.png")
    prediction_resized.save(save_path)
    print(f"Saved prediction mask: {save_path}")

# IOU 계산 및 예측된 마스크 저장 + 평균 IOU 계산
# IOU 계산 및 예측된 마스크 저장 + 평균 IOU 계산
# IOU 계산 및 예측된 마스크 저장 + 평균 IOU 계산
def test_and_save_predictions(model, test_loader, save_dir):
    model.eval()  # 평가 모드
    iou_scores = {'A': [], 'B': [], 'C': []}
    iou_threshold = 0.5

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # 저장 폴더 생성

    with torch.no_grad():
        for i, (images, masks, original_size, filename) in enumerate(test_loader):
            images = images.cuda()

            # 모델 추론
            outputs = model(images)['out']
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            # 예측 마스크를 원본 크기로 리사이즈
            pred_mask_np = predictions[0]  # 예측된 마스크 (512x512)
            prediction_resized = Image.fromarray(pred_mask_np.astype(np.uint8) * 255)
            prediction_resized = prediction_resized.resize(original_size, Image.NEAREST)  # 원본 크기(1080x1920 등)

            # 예측된 마스크를 다시 numpy 배열로 변환
            pred_mask_np_resized = np.array(prediction_resized) // 255  # 255를 나누어 이진화

            # 실제 마스크와 비교 (1080x1920 크기)
            true_mask_np = masks.squeeze(0).squeeze(0).cpu().numpy()  # 실제 마스크

            # IOU 계산
            iou = calculate_iou(pred_mask_np_resized, true_mask_np)
            status = "적합" if iou >= iou_threshold else "부적합"

            # A, B, C별로 IOU 저장
            if filename[0].startswith('A'):  # filename이 튜플이므로 [0]으로 접근
                iou_scores['A'].append(iou)
            elif filename[0].startswith('B'):
                iou_scores['B'].append(iou)
            elif filename[0].startswith('C'):
                iou_scores['C'].append(iou)

            # 예측 마스크 저장
            save_prediction_mask(pred_mask_np_resized, original_size, filename[0], save_dir)  # filename[0] 사용

            # IOU 결과 출력
            print(f"Image {filename[0]}: IOU = {iou:.4f}, {status}")

    # A, B, C별 평균 IOU 출력
    for key in ['A', 'B', 'C']:
        if iou_scores[key]:
            avg_iou = np.mean(iou_scores[key])
            print(f"Average IOU for {key}: {avg_iou:.4f}")
        else:
            print(f"No images for {key} to calculate IOU.")


# 테스트 진행 및 예측된 마스크 저장
test_and_save_predictions(model, test_loader, prediction_save_dir)

