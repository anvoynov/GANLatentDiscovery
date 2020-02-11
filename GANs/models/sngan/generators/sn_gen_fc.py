import numpy as np
from torch import nn
from ...generator import Generator

WIDTH = 128


def make_sn_fc_path_generator(latent_dim=128, img_size=28, image_channels=1, leak=0.0):
    def make_block(in_feat, out_feat):
        return nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.LeakyReLU(leak)
        )
    img_shape = [image_channels, img_size, img_size]
    model = nn.Sequential(
        make_block(latent_dim, WIDTH),
        nn.BatchNorm1d(WIDTH),
        make_block(WIDTH, 2 * WIDTH),
        nn.BatchNorm1d(2 * WIDTH),
        make_block(2 * WIDTH, 4 * WIDTH),
        nn.BatchNorm1d(4 * WIDTH),
        make_block(4 * WIDTH, 8 * WIDTH),
        nn.BatchNorm1d(8 * WIDTH),
        nn.Linear(8 * WIDTH, int(np.prod(img_shape))),
        nn.Tanh()
    )

    return Generator(model, latent_dim, img_shape)
