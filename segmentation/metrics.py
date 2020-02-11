import torch
from torchvision.transforms import ToPILImage, ToTensor, Resize

from torch_tools.utils import wrap_with_tqdm


def resize(m, target_shape):
    m = ToPILImage()(m.cpu().to(torch.float32))
    m = Resize(target_shape)(m)
    m = ToTensor()(m)
    return m.cuda()


def IoU(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


def MAE(mask1, mask2):
    diff = mask1.to(torch.float32) - mask2.to(torch.float32)
    return torch.mean(torch.abs(diff)).item()


def model_metrics(segmetation_model, dataloder, n_steps=None, stats=(IoU, MAE)):
    avg_values = [0.0 for _ in stats]

    step = 0
    for step, (img, mask) in wrap_with_tqdm(enumerate(dataloder), total=len(dataloder)):
        with torch.no_grad():
            img, mask = img.cuda(), mask.cuda()

            logits = segmetation_model(img)
            mask_prediction = torch.argmax(logits, dim=1, keepdim=False)

            if img.shape[-2:] != mask.shape[-2:]:
                print('resizing: {} -> {}'.format(img.shape[-2:], mask.shape[-2:]))
                mask = resize(mask, img.shape[-2:])

        for i, stat in enumerate(stats):
            avg_values[i] += stat(mask, mask_prediction)

        step += 1
        if n_steps is not None and step >= n_steps:
            break

    out_dict = {stat.__name__: avg / step for stat, avg in zip(stats, avg_values)}
    return out_dict
