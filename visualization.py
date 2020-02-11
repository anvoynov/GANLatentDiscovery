import numpy as np
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image
import io

from torch_tools.visualization import to_image

from utils import make_noise, one_hot


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=True):
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        if deformator is not None:
            z_deformed = z + deformator(one_hot(z.shape[1:], shift, dim).cuda())
        else:
            z_deformed = z + one_hot(z.shape[1:], shift, dim).cuda()
        shifted_image = G(z_deformed).cpu()[0]
        if shift == 0.0 and with_central_border:
            shifted_image = add_border(shifted_image)

        shifted_images.append(shifted_image)
    return shifted_images


def add_border(tensor):
    border = 3
    for ch in range(tensor.shape[0]):
        color = 1.0 if ch == 0 else -1
        tensor[ch, :border, :] = color
        tensor[ch, -border:,] = color
        tensor[ch, :, :border] = color
        tensor[ch, :, -border:] = color
    return tensor


@torch.no_grad()
def make_interpolation_chart(G, deformator=None, z=None,
                             shifts_r=10, shifts_count=5,
                             dims=None, dims_count=10, **kwargs):
    with_deformation = deformator is not None
    if with_deformation:
        deformator_is_training = deformator.training
        deformator.eval()
    z = z if z is not None else make_noise(1, G.dim_z).cuda()

    if with_deformation:
        original_img = G(z).cpu()
    else:
        original_img = G(z).cpu()

    imgs = []
    if dims is None:
        dims = range(dims_count)
    for i in dims:
        imgs.append(interpolate(G, z, shifts_r, shifts_count, i, deformator))

    rows_count = len(imgs) + 1
    fig, axs = plt.subplots(rows_count, **kwargs)

    axs[0].axis('off')
    axs[0].imshow(to_image(original_img, True))
    for ax, shifts_imgs, dim in zip(axs[1:], imgs, dims):
        ax.axis('off')
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1)), True))
        ax.text(0, 0, str(dim), fontsize=4)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig
