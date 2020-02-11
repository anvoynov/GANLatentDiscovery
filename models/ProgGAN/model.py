#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This work is based on the Theano/Lasagne implementation of
Progressive Growing of GANs paper from tkarras:
https://github.com/tkarras/progressive_growing_of_gans

PyTorch Model definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class PixelNormLayer(nn.Module):
    def __init__(self):
        super(PixelNormLayer, self).__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


class WScaleLayer(nn.Module):
    def __init__(self, size):
        super(WScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.randn([1]))
        self.b = nn.Parameter(torch.randn(size))
        self.size = size

    def forward(self, x):
        x_size = x.size()
        x = x * self.scale + self.b.view(1, -1, 1, 1).expand(
            x_size[0], self.size, x_size[2], x_size[3])

        return x


class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class NormUpscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(NormUpscaleConvBlock, self).__init__()
        self.norm = PixelNormLayer()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.wscale = WScaleLayer(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        x = self.conv(x)
        x = F.leaky_relu(self.wscale(x), negative_slope=0.2)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.features = nn.Sequential(
            NormConvBlock(512, 512, kernel_size=4, padding=3),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 512, kernel_size=3, padding=1),
            NormConvBlock(512, 512, kernel_size=3, padding=1),
            NormUpscaleConvBlock(512, 256, kernel_size=3, padding=1),
            NormConvBlock(256, 256, kernel_size=3, padding=1),
            NormUpscaleConvBlock(256, 128, kernel_size=3, padding=1),
            NormConvBlock(128, 128, kernel_size=3, padding=1),
            NormUpscaleConvBlock(128, 64, kernel_size=3, padding=1),
            NormConvBlock(64, 64, kernel_size=3, padding=1),
            NormUpscaleConvBlock(64, 32, kernel_size=3, padding=1),
            NormConvBlock(32, 32, kernel_size=3, padding=1),
            NormUpscaleConvBlock(32, 16, kernel_size=3, padding=1),
            NormConvBlock(16, 16, kernel_size=3, padding=1))

        self.output = nn.Sequential(OrderedDict([
                        ('norm', PixelNormLayer()),
                        ('conv', nn.Conv2d(16,
                                           3,
                                           kernel_size=1,
                                           padding=0,
                                           bias=False)),
                        ('wscale', WScaleLayer(3))
                    ]))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x
