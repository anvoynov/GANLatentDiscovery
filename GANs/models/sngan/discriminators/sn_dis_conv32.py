# DCGAN-like generator and discriminator
from torch import nn
from ..spectral_normalization import SpectralNorm


d_leak = 0.1
w_g = 4


class SNDiscriminator(nn.Module):
    def __init__(self):
        super(SNDiscriminator, self).__init__()

        self.conv1 = SpectralNorm(nn.Conv2d(1, 64, 3, stride=1, padding=(1,1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1,1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1,1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1,1)))

        self.fc = SpectralNorm(nn.Linear(w_g * w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(d_leak)(self.conv1(m))
        m = nn.LeakyReLU(d_leak)(self.conv2(m))
        m = nn.LeakyReLU(d_leak)(self.conv3(m))
        m = nn.LeakyReLU(d_leak)(self.conv4(m))
        m = nn.LeakyReLU(d_leak)(self.conv5(m))
        m = nn.LeakyReLU(d_leak)(self.conv6(m))
        m = nn.LeakyReLU(d_leak)(self.conv7(m))

        return self.fc(m.view(-1, w_g * w_g * 512))
