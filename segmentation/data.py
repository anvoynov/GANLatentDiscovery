from torch.utils.data import Dataset
from torchvision import transforms
from torch_tools.data import UnannotatedDataset, TransformedDataset


def central_crop(x):
    dims = x.size
    crop = transforms.CenterCrop(min(dims[0], dims[1]))
    return crop(x)


def _id(x):
    return x


class SegmentationDataset(Dataset):
    def __init__(self, images_root, masks_root, crop=True, size=None, mask_thr=0.5):
        self.mask_thr = mask_thr
        images_ds = UnannotatedDataset(images_root, transform=None)
        masks_ds = UnannotatedDataset(masks_root, transform=None)
        masks_ds.align_names(images_ds.img_files)

        resize = transforms.Compose([
            central_crop if crop else _id,
            transforms.Resize(size) if size is not None else _id,
            transforms.ToTensor()])
        shift_to_zero = lambda x: 2 * x - 1
        self.images_ds = TransformedDataset(images_ds, transforms.Compose([resize, shift_to_zero]))
        self.masks_ds = TransformedDataset(masks_ds, resize)

    def __len__(self):
        return len(self.images_ds)

    def __getitem__(self, index):
        mask = self.masks_ds[index] >= self.mask_thr
        return (self.images_ds[index], mask[0])
