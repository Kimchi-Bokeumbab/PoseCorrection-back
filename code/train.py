import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.posture_dataset import PostureDataset
from model.posture_classifier import PostureClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# 데이터 로더
train_dataset = PostureDataset("code/data/dataset/posture_data_merged.csv", mode='train')
val_dataset = PostureDataset("code/data/dataset/posture_data_merged.csv", mode='val')
test_dataset = PostureDataset("code/data/dataset/posture_data_merged.csv", mode='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 모델, 손실 함수, 옵티마이저
model = PostureClassifier(input_size=14, num_classes=5)  # 7점 × (x, y)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 학습
for epoch in range(EPOCHS):
    model.train()
    train_loss, correct = 0, 0
    for X, y in train_loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
    train_acc = correct / len(train_loader.dataset)

    # 검증
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == y).sum().item()
    val_acc = val_correct / len(val_loader.dataset)

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}")

# 테스트
model.eval()
test_correct = 0
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        test_correct += (outputs.argmax(1) == y).sum().item()
test_acc = test_correct / len(test_loader.dataset)
print(f"✅ Test Accuracy: {test_acc:.2f}")


# 테스트 결과 수집
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for X, y in test_loader:
        outputs = model(X)
        preds = outputs.argmax(1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
label_names = train_dataset.labels  # ['good_posture', 'shoulder_tilt', ...]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Classification Report
print("📋 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_names))

# 모델 저장
MODEL_PATH = "code/model/posture_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"모델이 저장되었습니다: {MODEL_PATH}")
