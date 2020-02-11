from torch import nn
from ..spectral_normalization import SpectralNorm


class SNFCDiscriminator(nn.Module):
    def __init__(self, img_size=28, image_channels=1):
        super(SNFCDiscriminator, self).__init__()

        self.model = nn.Sequential(
            SpectralNorm(nn.Linear(image_channels * img_size * img_size, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(256, 1)),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
