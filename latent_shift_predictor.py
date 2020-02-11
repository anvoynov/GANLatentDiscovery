import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import numpy as np


def save_hook(module, input, output):
    setattr(module, 'output', output)


class ResNetShiftPredictor(nn.Module):
    def __init__(self, dim, downsample=None):
        super(ResNetShiftPredictor, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            6, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)
        self.downsample = downsample

        # half dimension as we expect the model to be symmetric
        self.type_estimator = nn.Linear(512, np.product(dim))
        self.shift_estimator = nn.Linear(512, 1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        if self.downsample is not None:
            x1, x2 = F.interpolate(x1, self.downsample), F.interpolate(x2, self.downsample)
        self.features_extractor(torch.cat([x1, x2], dim=1))
        features = self.features.output.view([batch_size, -1])

        logits = self.type_estimator(features)
        shift = self.shift_estimator(features)

        return logits, shift.squeeze()


class LeNetShiftPredictor(nn.Module):
    def __init__(self, dim, channels=3, width=2):
        super(LeNetShiftPredictor, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(channels * 2, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_logits = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, np.product(dim))
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.convnet(torch.cat([x1, x2], dim=1))
        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        logits = self.fc_logits(features)
        shift = self.fc_shift(features)

        return logits, shift.squeeze()
