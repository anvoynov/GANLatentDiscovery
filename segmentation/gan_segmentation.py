from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from enum import Enum
from utils import one_hot, make_noise, run_in_background
from utils.prefetch_generator import background


class MaskSynthesizing(Enum):
    DIFF = 1
    INTENSITY = 2


def get_diff(img1, img2):
    return torch.mean(img2, dim=1, keepdim=True) - torch.mean(img1, dim=1, keepdim=True)


def normalize_per_sample(x, inplace=True):
    if not inplace:
        x = x.clone()
    for i in range(x.shape[0]):
        M, m = x[i].max(), x[i].min()
        x[i] = (x[i] - m) / (M - m)
    return x


def diff_to_mask(diff, thr):
    mask = torch.empty_like(diff)
    for i in range(mask.shape[0]):
        mask[i] = diff[i] < thr
    return mask.to(torch.long).squeeze()


class MaskGenerator(nn.Module):
    def __init__(self, G, deformator, background_dim, params, zs=None):
        super(MaskGenerator, self).__init__()
        self.G = G
        self.deformator = deformator
        self.background_dim = background_dim
        self.p = params
        self.zs = zs

    @torch.no_grad()
    def make_noise(self, batch_size, device):
        if self.zs is None:
            return make_noise(batch_size, self.G.dim_z, self.p.truncation).to(device)
        else:
            indices = torch.randint(0, len(self.zs), [batch_size], dtype=torch.long)
            z = self.zs[indices].to(device)
            return z


    @torch.no_grad()
    def gen_samples(self, z=None, classes=None, batch_size=None, device='cpu'):
        assert (z is None) ^ (batch_size is None), 'one of: z, batch_size should be provided'

        if z is None:
            z = self.make_noise(batch_size, device)
        if classes is None:
            classes = self.G.mixed_classes(z.shape[0]).to(device)

        shift = self.deformator(
            one_hot(self.G.dim_z, self.p.latent_shift_r, self.background_dim).to(device))

        img = self.G(z, classes)
        img_shifted_pos = self.G(z + shift, classes)

        if self.p.synthezing == MaskSynthesizing.DIFF:
            img_shifted_neg = self.G(z - shift, classes)
            diff = get_diff(0.5 * img_shifted_neg + 0.5, 0.5 * img_shifted_pos + 0.5)
            diff = normalize_per_sample(diff)
            mask = diff_to_mask(diff, self.p.mask_thr)

        elif self.p.synthezing == MaskSynthesizing.INTENSITY:
            intensity = 0.5 * torch.mean(img_shifted_pos, dim=1) + 0.5
            mask = (intensity < self.p.mask_thr).to(torch.long)

        return img, img_shifted_pos, mask

    @torch.no_grad()
    def filter_by_area(self, img_batch, img_pos_batch, ref_batch):
        if self.p.mask_size_low > 0.0 or self.p.mask_size_up < 1.0:
            ref_size = ref_batch.shape[-2] * ref_batch.shape[-1]
            ref_fraction = ref_batch.sum(dim=[-1, -2]).to(torch.float) / ref_size
            mask = (ref_fraction > self.p.mask_size_low) & (ref_fraction < self.p.mask_size_up)
            if torch.all(~mask):
                return None
            img_batch, img_pos_batch, ref_batch = \
                img_batch[mask], img_pos_batch[mask], ref_batch[mask]
        return img_batch, img_pos_batch, ref_batch

    @torch.no_grad()
    def forward(self, max_retries=100, z=None, classes=None, return_steps=False):
        img, ref = None, None
        step = 0
        device = self.G.target_classes.device
        while img is None or img.shape[0] < self.p.batch_size:
            step += 1
            if step > max_retries:
                raise Exception('generator was disable to synthesize mask')

            if z is None or step > 1:
                z = self.make_noise(self.p.batch_size, device)
                classes = self.G.mixed_classes(self.p.batch_size).to(device)
            z = self.augment_z(z)

            img_batch, img_pos_batch, ref_batch = \
                self.gen_samples(z=z, classes=classes, device=device)

            # filtration
            mask_area_filtration = self.filter_by_area(img_batch, img_pos_batch, ref_batch)
            if mask_area_filtration is not None:
                img_batch, img_pos_batch, ref_batch = mask_area_filtration
            else:
                continue

            # batch update
            if img is None:
                img, ref = img_batch, ref_batch
            else:
                img = torch.cat([img, img_batch])[:self.p.batch_size]
                ref = torch.cat([ref, ref_batch])[:self.p.batch_size]

        if return_steps:
            return img, ref, step
        return img, ref


def it_mask_gen(mask_gen, out_device='cpu'):
    while True:
        img, ref = mask_gen()
        yield img.to(out_device), ref.to(out_device)
