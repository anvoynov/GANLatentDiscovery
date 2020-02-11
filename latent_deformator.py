import torch
from torch import nn
from torch.nn import functional as F
from enum import Enum
import numpy as np

from ortho_utils import torch_expm


class DeformatorType(Enum):
    FC = 1
    LINEAR = 2
    ID = 3
    ORTHO = 4
    PROJECTIVE = 5
    RANDOM = 6



class LatentDeformator(nn.Module):
    def __init__(self, target_dim, inner_dim=1024, out_dim=None,
                 type=DeformatorType.FC, random_init=False):
        super(LatentDeformator, self).__init__()
        self.type = type
        self.input_dim = np.product(target_dim)
        self.out_dim = out_dim
        if out_dim is None:
            self.out_dim = self.input_dim

        if self.type == DeformatorType.FC:
            self.fc1 = nn.Linear(self.input_dim, inner_dim)
            self.bn1 = nn.BatchNorm1d(inner_dim)
            self.act1 = nn.ELU()

            self.fc2 = nn.Linear(inner_dim, inner_dim)
            self.bn2 = nn.BatchNorm1d(inner_dim)
            self.act2 = nn.ELU()

            self.fc3 = nn.Linear(inner_dim, inner_dim)
            self.bn3 = nn.BatchNorm1d(inner_dim)
            self.act3 = nn.ELU()

            self.fc4 = nn.Linear(inner_dim, self.out_dim)

        elif self.type in [DeformatorType.LINEAR, DeformatorType.PROJECTIVE]:
            self.linear = nn.Linear(self.input_dim, self.out_dim)
            self.linear.weight.data = torch.zeros_like(self.linear.weight.data)

            min_dim = int(min(self.input_dim, self.out_dim))
            self.linear.weight.data[:min_dim, :min_dim] = torch.eye(min_dim)
            if random_init:
                self.linear.weight.data += 0.1 * torch.randn_like(self.linear.weight.data)

        elif self.type == DeformatorType.ORTHO:
            assert self.input_dim == self.out_dim, 'In/out dims must be equal for ortho'
            self.log_mat_half = nn.Parameter((1.0 if random_init else 0.001) * torch.randn(
                [self.input_dim, self.input_dim], device='cuda'), True)

        elif self.type == DeformatorType.RANDOM:
            self.linear = torch.empty([self.out_dim, self.input_dim])
            nn.init.orthogonal_(self.linear)

    def forward(self, input):
        original_shape = input.shape
        if self.type == DeformatorType.ID:
            return input

        input = input.view([-1, self.input_dim])
        if self.type == DeformatorType.FC:
            x1 = self.fc1(input)
            x = self.act1(self.bn1(x1))

            x2 = self.fc2(x)
            x = self.act2(self.bn2(x2 + x1))

            x3 = self.fc3(x)
            x = self.act3(self.bn3(x3 + x2 + x1))

            out = self.fc4(x) + input
        elif self.type == DeformatorType.LINEAR:
            out  = self.linear(input)
        elif self.type == DeformatorType.PROJECTIVE:
            input_norm = torch.norm(input, dim=1, keepdim=True)
            out = self.linear(input)
            out = (input_norm / torch.norm(out, dim=1, keepdim=True)) * out
        elif self.type == DeformatorType.ORTHO:
            mat = torch_expm((self.log_mat_half - self.log_mat_half.transpose(0, 1)).unsqueeze(0))
            out = F.linear(input, mat)
        elif self.type == DeformatorType.RANDOM:
            self.linear = self.linear.to(input.device)
            out = F.linear(input, self.linear)

        return out.view(original_shape)


def normal_projection_stat(x):
    x = x.view([x.shape[0], -1])
    direction = torch.randn(x.shape[1], requires_grad=False, device=x.device)
    direction = direction / torch.norm(direction)
    projection = torch.matmul(x, direction)

    std, mean = torch.std_mean(projection)
    return std, mean
