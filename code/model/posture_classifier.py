import torch.nn as nn

class PostureClassifier(nn.Module):
    def __init__(self, input_size=14, num_classes=5):
        super(PostureClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)
