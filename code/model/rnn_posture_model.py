import torch.nn as nn

class RNNPostureModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=64, num_layers=1, num_classes=5):
        super(RNNPostureModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden)
        out = out[:, -1, :]    # 마지막 타임스텝의 출력만 사용
        out = self.fc(out)
        return out
