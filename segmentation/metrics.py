import torch

from torchvision.transforms import ToPILImage, ToTensor, Resize
from torch_tools.utils import wrap_with_tqdm


def resize(x, target_shape):
    x = ToPILImage()(x.cpu().to(torch.float32))
    x = Resize(target_shape)(x)
    x = ToTensor()(x)
    return x.cuda()


def MAE(mask1, mask2):
    diff = mask1.to(torch.float32) - mask2.to(torch.float32)
    return torch.mean(torch.abs(diff)).item()


def model_metrics(segmetation_model, dataloder, n_steps=None, stats=(MAE,)):
    avg_values = {}
    out_dict = {}

    n_steps = len(dataloder) if n_steps is None else n_steps
    step = 0
    for step, (img, mask) in wrap_with_tqdm(enumerate(dataloder), total=n_steps):
        with torch.no_grad():
            img, mask = img.cuda(), mask.cuda()

        if img.shape[-2:] != mask.shape[-2:]:
            mask = resize(mask, img.shape[-2:])

        prediction = segmetation_model(img)

        for metric in stats:
            method = metric.__name__
            if method not in avg_values:
                avg_values[method] = 0.0

            avg_values[method] += metric(mask, prediction)

        step += 1
        if n_steps is not None and step >= n_steps:
            break

    for metric in stats:
        method = metric.__name__
        out_dict[method] = avg_values[method] / step

    return out_dict
