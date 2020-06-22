import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset


class PseudoLabelDataset(IterableDataset):
    def __init__(self, G, dim, r=10.0, batch_size=32, deformator=None, size=None):
        super(PseudoLabelDataset, self).__init__()
        self.G = G
        self.deformator = deformator
        self.r = r
        self.batch = batch_size
        self.dim = dim
        self.size = size

    @torch.no_grad()
    def __iter__(self):
        while True:
            latent_shape = [self.batch, self.G.dim_z] if type(self.G.dim_z) == int else \
                [self.batch] + self.G.dim_z

            z = torch.randn(latent_shape, device='cuda')
            signs = torch.randint(0, 2, [self.batch], device='cuda')

            shifts = torch.zeros(latent_shape, device='cuda')
            shifts[:, self.dim] = self.r * (2.0 * signs - 1.0)
            if self.deformator is not None:
                shifts = self.deformator(shifts)

            with torch.no_grad():
                img = self.G(z + shifts).detach()

            if self.size is not None:
                img = F.interpolate(img, self.size)
                torch.cuda.empty_cache()

            yield img, signs


class ModelLabeledDataset(Dataset):
    def __init__(self, ds, model):
        super(ModelLabeledDataset, self).__init__()
        self.ds = ds
        self.model = model

    def __len__(self):
        return len(self.ds)

    @torch.no_grad()
    def __getitem__(self, item):
        img = self.ds[item]
        label = torch.argmax(self.model(img.unsqueeze(0)), dim=1).squeeze()

        return img, label
