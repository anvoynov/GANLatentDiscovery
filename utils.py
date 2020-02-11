import torch


def make_noise(batch, dim):
    if isinstance(dim, int):
        dim = [dim]
    return torch.randn([batch] + dim)


def one_hot(dims, value, indx):
    vec = torch.zeros(dims)
    vec[indx] = value
    return vec
