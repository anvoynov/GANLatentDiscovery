import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from .data import UnannotatedDataset


def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


class SamplesGrid(object):
    def __init__(self, dataset_dir, size):
        self.dataset_dir = dataset_dir
        self.set_size(size)

    def __call__(self):
        grid = make_grid(next(iter(self.dataloader)), nrow=self.grid_size[0])
        return to_image(grid)

    def set_size(self, size):
        self.grid_size = size
        self.dataloader = torch.utils.data.DataLoader(
            UnannotatedDataset(self.dataset_dir), size[0] * size[1], shuffle=True)
