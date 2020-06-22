import torch
from torchvision.transforms import ToPILImage, ToTensor, Resize


def resize(x, target_shape):
    x = ToPILImage()(x.cpu().to(torch.float32))
    x = Resize(target_shape)(x)
    x = ToTensor()(x)
    return x.cuda()


def resize_min_edge(x, size):
    img_shape = x.shape[-2:]
    if img_shape[0] > img_shape[1]:
        x = resize((x[0] + 1.0) / 2.0, (size * img_shape[0] // img_shape[1], size))
    else:
        x = resize((x[0] + 1.0) / 2.0, (size, size * img_shape[1] // img_shape[0]))

    return x.unsqueeze(0) * 2.0 - 1.0


class SegmentationInference(object):
    def __init__(self, model=None, resize_to=None):
        self.model = model
        self.resize_to = resize_to

    @torch.no_grad()
    def __call__(self, img, mask=None):
        return self.apply(img, mask=None)

    @torch.no_grad()
    def apply(self, img, mask=None):
        img_shape = img.shape[-2:]
        if mask is None:
            if self.model is not None:
                if self.resize_to is not None:
                    img = resize_min_edge(img, self.resize_to)
                mask = self.model(img)
            else:
                raise Exception(
                    'Either both (img, mask) should be provided or self.model is not None')

        if  len(mask.shape) == 4:
            mask = torch.sigmoid(mask)[:, 1]

        if self.resize_to is not None and mask.shape[-2:] != img_shape:
            mask = resize(mask, img_shape)

        return mask


class Threshold(SegmentationInference):
    def __init__(self, model=None, thr=0.5, resize_to=None):
        super(Threshold, self).__init__(model, resize_to)
        self.thr = thr

    @torch.no_grad()
    def __call__(self, img, mask=None):
        mask = self.apply(img, mask)
        return mask >= self.thr
