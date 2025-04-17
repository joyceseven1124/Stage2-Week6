import torch.nn as nn

class MyClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MyClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 30),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(30, 9)
        )

    def forward(self, x):
        return self.layers(x)
