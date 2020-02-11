from torch import nn


class Crop(nn.Module):
    def __init__(self, crop):
        super(Crop, self).__init__()
        self.crop = crop

    def forward(self, input):
        return input[:, :, :-self.crop[0], :-self.crop[1]]


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return input.view(self.target_shape)


class Identity(nn.Module):
    def __init__(self, *args):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input[0]
