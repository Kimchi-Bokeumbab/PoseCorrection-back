from data.posture_dataset import PostureDataset
from model.posture_classifier import PostureClassifier
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

dataset = PostureDataset('code/data/dataset/posture_data_merged.csv')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = PostureClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    total_loss = 0
    for coords, labels in train_loader:
        outputs = model(coords)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    # 학습된 모델 저장
torch.save(model.state_dict(), 'posture_model.pth')
