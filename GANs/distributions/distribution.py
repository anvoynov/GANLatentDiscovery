from torch import nn


class BaseDistribution(nn.Module):
    def __init__(self, dim, device='cuda'):
        super(BaseDistribution, self).__init__()
        self.device = device
        self.dim = dim

    def cuda(self, device=None):
        super(BaseDistribution, self).cuda(device)
        self.device = 'cuda' if device is None else device

    def cpu(self):
        super(BaseDistribution, self).cpu()
        self.device='cpu'

    def to(self, device):
        super(BaseDistribution, self).to(device)
        self.device = device

    def forward(self, batch_size):
        raise NotImplementedError
