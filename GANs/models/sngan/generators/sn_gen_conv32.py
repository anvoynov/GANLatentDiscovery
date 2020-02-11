# DCGAN-like generator and discriminator
from torch import nn

from ....utils import Reshape
from ...generator import Generator


def make_sn_path_generator(latent_dim=128, img_size=64, leak=0.0, seed_dim=4, first_linear=True):
    def make_upconv_layer(in_channels, out_channels, kernel, stride, padding=(0, 0)):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
        )

    if first_linear:
        first_layer = [
            nn.Linear(latent_dim, seed_dim * seed_dim * 512),
            Reshape([-1, 512, seed_dim, seed_dim])
        ]
    else:
        first_layer = [make_upconv_layer(latent_dim, 512, 4, 1)]

    model = nn.Sequential(*(
        first_layer +
        [
            nn.LeakyReLU(leak),
            make_upconv_layer(512, 256, 4, 2, (1, 1)),
            nn.LeakyReLU(leak),
            make_upconv_layer(256, 128, 4, 2, (1, 1)),
            nn.LeakyReLU(leak),
            make_upconv_layer(128, 64, 4, 2, (1, 1)),
            nn.LeakyReLU(leak),
            make_upconv_layer(64, 1, 3, 1, (1, 1)),
            nn.Tanh()
        ]))

    latent_dim = [latent_dim] if first_linear else [latent_dim, 1, 1]
    return Generator(model, latent_dim, [1, img_size, img_size])
