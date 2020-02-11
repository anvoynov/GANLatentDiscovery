# fork of https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
import torch
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, target='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.target = target
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.target + "_u")
        v = getattr(self.module, self.target + "_v")
        w = getattr(self.module, self.target + "_bar")

        size = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(size, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(size, -1).data, v.data))

        sigma = u.dot(w.view(size, -1).mv(v))
        setattr(self.module, self.target, w / sigma.expand_as(w))

    def _made_params(self):
        return hasattr(self.module, '{}_u'.format(self.target)) and \
               hasattr(self.module, '{}_v'.format(self.target)) and \
               hasattr(self.module, '{}_bar'.format(self.target))

    def _make_params(self):
        w = getattr(self.module, self.target)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.target]

        self.module.register_parameter(self.target + "_u", u)
        self.module.register_parameter(self.target + "_v", v)
        self.module.register_parameter(self.target + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
