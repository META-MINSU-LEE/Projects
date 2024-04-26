import pickle
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional Layer 그룹 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        # Convolutional Layer 그룹 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Convolutional Layer 그룹 3
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Convolutional Layer 그룹 4
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Spatial Dropout
        self.dropout = nn.Dropout2d(p=0.2)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        # self.flatten = nn.Linear(128*6*6, 4608)
        self.dense = nn.Linear(4608, 512)
        # 출력층 크기를 9
        self.out = nn.Linear(512, 9)

    def forward(self, x):
        # Conv-Pool-Conv 그룹 1
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)

        # Conv-Pool-Conv 그룹 2
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)

        # Conv-Pool-Conv 그룹 3
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.conv6(x)

        # Conv-Pool-Conv 그룹 4
        x = self.conv7(x)
        x = self.pool4(x)
        x = self.conv8(x)

        # Spatial Dropout
        x = self.dropout(x)
        x = self.pool5(x)

        # Fully Connected Layers
        x = nn.ReLU()(self.flatten(x))
        x = nn.ReLU()(self.dense(x))
        # Softmax 출력층
        x = nn.Softmax(dim=1)(self.out(x))

        return x


# # 모델 인스턴스 생성
# model = CNN()
#
# # Adam 최적화 사용
# optimizer = torch.optim.Adam(model.parameters())

output_folder = './augmented_images'

# 배치 크기와 에포크(epoch) 수
batch_size = 100
epochs = 20

learning_rate = 0.001

# 데이터 전처리 및 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.ToTensor(),         # 텐서로 변환
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 정규화
])

# 데이터셋 생성
dataset = datasets.ImageFolder(root=output_folder, transform=transform)

class_samples = {}

for i in range(len(dataset)):
    _, label = dataset[i]
    if label not in class_samples:
        class_samples[label] = []
    class_samples[label].append(i)

# 클래스 간의 샘플 수가 동일하도록 분할
train_samples = []
val_samples = []
test_samples = []
for label, samples in class_samples.items():
    n_samples = len(samples)
    n_train = int(0.65 * n_samples)
    n_val = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_val

    # 클래스 별로 분할된 샘플 추가
    train_samples.extend(samples[:n_train])
    val_samples.extend(samples[n_train:n_train + n_val])
    test_samples.extend(samples[n_train + n_val:])

# 클래스 별 샘플 수를 계산하고 균형 잡힌 데이터셋을 생성합니다.
class_counts = {label: len(samples) for label, samples in class_samples.items()}
min_samples = min(class_counts.values())

balanced_indices = []
for label, samples in class_samples.items():
    if len(samples) > min_samples:
        balanced_indices.extend(np.random.choice(samples, min_samples, replace=False))
    else:
        balanced_indices.extend(samples)

balanced_dataset = Subset(dataset, balanced_indices)
balanced_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)

# 분할된 샘플 인덱스를 사용하여 데이터셋 분할
train_dataset = Subset(dataset, train_samples)
val_dataset = Subset(dataset, val_samples)
test_dataset = Subset(dataset, test_samples)

# 분할된 데이터셋을 데이터 로더로 변환합니다.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# GPU 사용 가능 여부 확인
device = torch.device("cuda")

# 모델 인스턴스 생성
model = CNN().to(device)

print(torch.cuda.is_available())

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(model)

# 손실 함수 및 정확도 기록
train_losses = []
val_losses = []
accuracies = []

patience = 5  # 허용횟수
early_stopping_counter = 0  # 얼리스탑 카운터
best_val_loss = float('inf')  # 최고 검증 손실값

# 학습
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 테스트
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # 입력 데이터를 GPU로 이동
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 손실 함수 및 정확도 평균 계산
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    accuracies.append(correct / total)
    # 정확도 계산
    accuracy = accuracy_score(y_true, y_pred)
    # 정밀도 계산
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
    # 재현율 계산
    recall = recall_score(y_true, y_pred, average='weighted')
    # f1 스코어 계산
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Epoch [{epoch + 1}/{epochs}]")
    print(f"Train Loss: {running_loss / len(train_loader)}")
    print(f"Val Loss: {val_loss / len(val_loader)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # epoch이 절반 이상 진행됐을 때
    if epoch > epochs / 2:
        # early stop 조건 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 얼리스탑 카운터 초기화
        else:
            early_stopping_counter += 1

        # early stop 허용 횟수를 초과하면 학습 중단
        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

# 모델 평가를 위한 부분
model.eval()
test_loss = 0.0
correct = 0
total = 0
y_true_test = []
y_pred_test = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 위의 테스트 코드 다음에 균형 잡힌 데이터셋 평가 코드를 추가합니다.
y_true_balanced = []
y_pred_balanced = []
model.eval()
with torch.no_grad():
    for images, labels in balanced_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true_balanced.extend(labels.cpu().numpy())
        y_pred_balanced.extend(predicted.cpu().numpy())

precision_balanced = precision_score(y_true_balanced, y_pred_balanced, average=None)
recall_balanced = recall_score(y_true_balanced, y_pred_balanced, average=None)
f1_balanced = f1_score(y_true_balanced, y_pred_balanced, average=None)

print('Balanced Dataset Performance:')
print('Precision:', precision_balanced)
print('Recall:', recall_balanced)
print('F1 Score:', f1_balanced)

# 모델의 최종 성능 평가
model.eval()
test_loss = 0.0
correct = 0
total = 0
y_true_test = []
y_pred_test = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        test_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        y_true_test.extend(labels.cpu().numpy())
        y_pred_test.extend(predicted.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i in range(len(images)):
            print(f"예측결과: {predicted[i]}, 정답: {labels[i]}")

# 평가 지표 계산
test_accuracy = accuracy_score(y_true_test, y_pred_test)
test_precision = precision_score(y_true_test, y_pred_test, average='weighted', zero_division=1)
test_recall = recall_score(y_true_test, y_pred_test, average='weighted')
test_f1 = f1_score(y_true_test, y_pred_test, average='weighted')
print("모델의 최종 성능")

# 평가 결과 출력
print("Test Loss: {:.4f}".format(test_loss / len(test_loader)))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Test Precision: {:.2f}%".format(test_precision * 100))
print("Test Recall: {:.2f}%".format(test_recall * 100))
print("Test F1 Score: {:.2f}%".format(test_f1 * 100))
# 그래프 그리기
plt.figure(figsize=(10, 5))

# 손실 함수 그래프
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train')
plt.plot(range(1, epochs + 1), val_losses, label='Validation')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), accuracies)
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()

# 모델 평가 완료 후 y_true_test와 y_pred_test를 파일로 저장
with open('y_true_test.pkl', 'wb') as f:
    pickle.dump(y_true_test, f)

with open('y_pred_test.pkl', 'wb') as f:
    pickle.dump(y_pred_test, f)

# 레이블을 불러옵니다.
with open('y_true_test.pkl', 'rb') as f:
    y_true_test = pickle.load(f)

with open('y_pred_test.pkl', 'rb') as f:
    y_pred_test = pickle.load(f)

# 데이터셋 생성
dataset = datasets.ImageFolder(root=output_folder, transform=transform)
# 클래스 이름을 인덱스와 함께 가져옵니다.
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# 혼동 행렬을 생성합니다.
conf_matrix = confusion_matrix(y_true_test, y_pred_test)

# 혼동 행렬을 정규화하여 백분율로 변환합니다.
conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
conf_matrix_percentage *= 100

# 클래스 이름을 혼동 행렬의 레이블로 사용합니다.
labels = [idx_to_class[i] for i in range(len(idx_to_class))]

# 혼동 행렬을 시각화합니다.
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix_percentage, annot=True, fmt='.1f', cmap='viridis', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Percentage)')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# 클래스별 성능 지표를 계산합니다.
precision = precision_score(y_true_test, y_pred_test, average=None, zero_division=0)
recall = recall_score(y_true_test, y_pred_test, average=None)
f1 = f1_score(y_true_test, y_pred_test, average=None)

# 데이터셋에 대한 성능 지표를 계산하는 코드 부분 뒤에,
# 여기에 Balanced 및 Imbalanced 데이터셋의 성능 지표를 계산하는 코드를 추가합니다.
# 클래스 별 샘플 수를 출력합니다.
class_counts = {label: len(samples) for label, samples in class_samples.items()}
print('Class counts:', class_counts)

# 가장 적은 샘플 수를 가진 클래스의 샘플 수를 찾습니다.
min_samples = min(class_counts.values())

# 모든 클래스에서 min_samples 만큼 샘플을 선택합니다. (Balanced)
balanced_indices = []
for label, samples in class_samples.items():
    if len(samples) > min_samples:
        balanced_indices.extend(np.random.choice(samples, min_samples, replace=False))
    else:
        balanced_indices.extend(samples)

# Balanced 데이터셋에 대한 y_true, y_pred 생성을 위한 DataLoader 설정
balanced_dataset = Subset(dataset, balanced_indices)
balanced_loader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=False)

# Balanced 데이터셋에 대한 예측 수행
y_true_balanced = []
y_pred_balanced = []
model.eval()
with torch.no_grad():
    for images, labels in balanced_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true_balanced.extend(labels.cpu().numpy())
        y_pred_balanced.extend(predicted.cpu().numpy())

# Balanced 데이터셋에 대한 성능 지표 계산
precision_balanced = precision_score(y_true_balanced, y_pred_balanced, average=None)
recall_balanced = recall_score(y_true_balanced, y_pred_balanced, average=None)
f1_balanced = f1_score(y_true_balanced, y_pred_balanced, average=None)

print('Balanced Dataset:')
print('Precision:', precision_balanced)
print('Recall:', recall_balanced)
print('F1 Score:', f1_balanced)

# Imbalanced 데이터셋에 대한 성능 지표 계산 (원본 데이터셋 사용)
precision_imbalanced = precision_score(y_true_test, y_pred_test, average=None)
recall_imbalanced = recall_score(y_true_test, y_pred_test, average=None)
f1_imbalanced = f1_score(y_true_test, y_pred_test, average=None)

# 결과를 출력합니다.
print('Imbalanced Dataset:')
print('Precision:', precision_imbalanced)
print('Recall:', recall_imbalanced)
print('F1 Score:', f1_imbalanced)
