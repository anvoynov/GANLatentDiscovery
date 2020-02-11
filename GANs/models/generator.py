import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, model, out_img_shape, distribution):
        super(Generator, self).__init__()

        self.model = model
        self.out_img_shape = out_img_shape
        self.distribution = distribution
        self.force_no_grad = False

    def cuda(self, device=None):
        super(Generator, self).cuda(device)
        self.distribution.cuda()

    def forward(self, batch_size):
        if self.force_no_grad:
            with torch.no_grad():
                img = self.model(self.distribution(batch_size))
        else:
            img = self.model(self.distribution(batch_size))

        img = img.view(img.shape[0], *self.out_img_shape)
        return img
