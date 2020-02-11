import torch
from utils import one_hot


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


class MaskGenerator(object):
    def __init__(self, G, deformator, background_dim, r, diff_to_mask_thr, use_diff=False):
        self.G = G
        self.deformator = deformator
        self.background_dim = background_dim
        self.r = r
        self.mask_thr = diff_to_mask_thr
        self.use_diff = use_diff

    @torch.no_grad()
    def __call__(self, z=None, classes=None, batch_size=None):
        assert (z is None) ^ (batch_size is None), 'one of: z, batch_size should be provided'

        if z is None:
            z = torch.randn([batch_size, self.G.dim_z]).cuda()
        if classes is None:
            classes = self.G.mixed_classes(z.shape[0]).cuda()

        shift = self.deformator(one_hot(self.G.dim_z, self.r, self.background_dim).cuda())

        img_shifted_neg = self.G(z - 0.5 * shift, classes)
        img_shifted_pos = self.G(z + 0.5 * shift, classes)

        if self.use_diff:
            diff = get_diff(img_shifted_neg, img_shifted_pos)
            diff = normalize_per_sample(diff)
            mask = diff_to_mask(diff, self.mask_thr)
        else:
            intensity = torch.mean(torch.abs(img_shifted_pos), dim=1)
            mask = (intensity < self.mask_thr).to(torch.long)

        return img_shifted_neg, img_shifted_pos, mask
