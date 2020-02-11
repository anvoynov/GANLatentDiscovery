from torch import nn
import numpy as np
from GANs.utils import Identity
from ..spectral_normalization import SpectralNorm


SN_RES_DIS_CONFIGS = {
    'sn_resnet32': [[128, True],
                    [128, False],
                    [128, False],
                    [128,]],
    'sn_resnet48': [[64, True],
                    [128, True],
                    [256, True],
                    [512, True],
                    [1024,]],
    'sn_resnet64': [[64, True],
                    [2 * 64, True],
                    [4 * 64, True],
                    [8 * 64, True],
                    [16 * 64,]],
    'sn_resnet128': [[64, True],
                     [2 * 64, True],
                     [4 * 64, True],
                     [8 * 64, True],
                     [16 * 64, False],
                     [16 * 64,]],
    'sn_resnet256': [[64, True],
                     [2 * 64, True],
                     [4 * 64, True],
                     [8 * 64, True],
                     [8 * 64, True],
                     [16 * 64, False],
                     [16 * 64,]],
}
SN_RES_DIS_CONFIGS['sn_resnet32_corrupted'] = SN_RES_DIS_CONFIGS['sn_resnet32']


class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=2, padding=0) if downsample else Identity(),
        )

        self.bypass = Identity()
        if downsample:
            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=2, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class OptimizedResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResnetDiscriminator(nn.Module):
    def __init__(self, config, image_channels=3):
        super(ResnetDiscriminator, self).__init__()
        res_blocks = [OptimizedResBlockDiscriminator(image_channels, config[0][0])]
        res_blocks += [
            ResBlockDiscriminator(config[i][0], config[i + 1][0], config[i][1])
            for i in range(len(config) - 1)
        ]
        res_blocks.append(nn.ReLU())

        self.model = nn.Sequential(*res_blocks)
        self.fc = nn.Linear(config[-1][0], 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        features = self.model(x).mean(dim=[-2, -1])
        return self.fc(features)
