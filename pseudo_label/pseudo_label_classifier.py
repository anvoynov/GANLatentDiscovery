from torch import nn


class PseudoLabelClassifier(nn.Module):
    def __init__(self, out_dim=2, channels=3):
        super(PseudoLabelClassifier, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels, 6, kernel_size=(5, 5)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.BatchNorm2d(120),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, out_dim)
        )

    def forward(self, x):
        features = self.convnet(x)
        features = features.mean(dim=[-1, -2])
        features = features.view(x.shape[0], -1)

        return self.fc(features)
