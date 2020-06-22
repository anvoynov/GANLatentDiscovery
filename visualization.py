import numpy as np
import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from PIL import Image
import io
import os

from torch_tools.visualization import to_image

from utils import make_noise, one_hot


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)


@torch.no_grad()
def interpolate(G, z, shifts_r, shifts_count, dim, deformator=None, with_central_border=False):
    shifted_images = []
    for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
        if deformator is not None:
            latent_shift = deformator(one_hot(deformator.input_dim, shift, dim).cuda())
        else:
            latent_shift = one_hot(G.dim_shift, shift, dim).cuda()
        shifted_image = G.gen_shifted(z, latent_shift).cpu()[0]
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
                             shifts_r=10.0, shifts_count=5,
                             dims=None, dims_count=10, texts=None, **kwargs):
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

    if texts is None:
        texts = dims
    for ax, shifts_imgs, text in zip(axs[1:], imgs, texts):
        ax.axis('off')
        plt.subplots_adjust(left=0.5)
        ax.imshow(to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
        ax.text(-20, 21, str(text), fontsize=10)

    if deformator is not None and deformator_is_training:
        deformator.train()

    return fig


@torch.no_grad()
def inspect_all_directions(G, deformator, out_dir, zs=None, num_z=3, shifts_r=8.0):
    os.makedirs(out_dir, exist_ok=True)

    step = 20
    max_dim = G.dim_shift
    zs = zs if zs is not None else make_noise(num_z, G.dim_z).cuda()
    shifts_count = zs.shape[0]

    for start in range(0, max_dim - 1, step):
        imgs = []
        dims = range(start, min(start + step, max_dim))
        for z in zs:
            z = z.unsqueeze(0)
            fig = make_interpolation_chart(
                G, deformator=deformator, z=z,
                shifts_count=shifts_count, dims=dims, shifts_r=shifts_r,
                dpi=250, figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
            fig.canvas.draw()
            plt.close(fig)
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # crop borders
            nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
            img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
            imgs.append(img)

        out_file = os.path.join(out_dir, '{}_{}.jpg'.format(dims[0], dims[-1]))
        print('saving chart to {}'.format(out_file))
        Image.fromarray(np.hstack(imgs)).save(out_file)
