import torch.nn as nn

class PostureClassifier(nn.Module):
    def __init__(self):
        super(PostureClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.net(x)
