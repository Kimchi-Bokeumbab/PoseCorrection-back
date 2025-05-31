import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from data.posture_data import PostureDataset
from model.rnn_posture_model import RNNPostureModel

# Load dataset
dataset = PostureDataset("code/data/dataset/posture_chunk_data.csv")
label_encoder = dataset.get_label_encoder()
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model, Loss, Optimizer
model = RNNPostureModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Lists for tracking loss and accuracy
loss_history = []
acc_history = []

# Training
for epoch in range(20):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            predicted = pred.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    acc = correct / total

    # 기록
    loss_history.append(total_loss)
    acc_history.append(acc)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {acc:.2%}")

    # Save
    torch.save(model.state_dict(), "code/model/rnn_posture_model2.pth")
    print("Model saved to rnn_posture_model2.pth")

# 그래프 그리기
plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(loss_history, marker='o')
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(acc_history, marker='o')
plt.title("Validation Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.show()
