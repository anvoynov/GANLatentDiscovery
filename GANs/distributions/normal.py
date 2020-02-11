import torch
from ..distributions.distribution import BaseDistribution


class NormalDistribution(BaseDistribution):
    def __init__(self, dim):
        super(NormalDistribution, self).__init__(dim)

    def forward(self, batch_size):
        return torch.randn([batch_size, self.dim]).to(self.device)
