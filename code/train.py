import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
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
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Acc: {acc:.2%}")
