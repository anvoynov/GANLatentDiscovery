from torch import nn
import numpy as np

from GANs.models.generator import Generator
from GANs.models.sngan.spectral_normalization import SpectralNorm
from GANs.utils import Reshape
from GANs.distributions.normal import NormalDistribution


GEN_SIZE=64


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            SpectralNorm(self.conv1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SpectralNorm(self.conv2)
            )

        if in_channels == out_channels:
            self.bypass = nn.Upsample(scale_factor=2)
        else:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1))
            )
            nn.init.xavier_uniform_(self.bypass[1].weight.data, 1.0)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


def make_snresnet_generator(resnet_gen_config, img_size=128, channels=3,
                            distribution=NormalDistribution(128)):
    def make_dense():
        dense = nn.Linear(
            distribution.dim, resnet_gen_config.seed_dim**2 * resnet_gen_config.channels[0])
        nn.init.xavier_uniform_(dense.weight.data, 1.)
        return dense

    def make_final():
        final = nn.Conv2d(resnet_gen_config.channels[-1], channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(final.weight.data, 1.)
        return final

    model_channels = resnet_gen_config.channels

    input_layers = [
        SpectralNorm(make_dense()),
        Reshape([-1, model_channels[0], 4, 4])
    ]
    res_blocks = [
        ResBlockGenerator(model_channels[i], model_channels[i + 1])
        for i in range(len(model_channels) - 1)
    ]
    out_layers = [
        nn.BatchNorm2d(model_channels[-1]),
        nn.ReLU(inplace=True),
        SpectralNorm(make_final()),
        nn.Tanh()
    ]

    model = nn.Sequential(*(input_layers + res_blocks + out_layers))

    return Generator(model, [channels, img_size, img_size], distribution)
