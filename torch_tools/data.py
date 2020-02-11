import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .utils import numerical_order, wrap_with_tqdm, make_verbose


def _filename(path):
    return os.path.basename(path).split('.')[0]


def imagenet_transform(size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor(),
        normalize,])


class UnannotatedDataset(Dataset):
    def __init__(self, root_dir, sorted=False,
                 transform=transforms.Compose(
                     [
                         transforms.ToTensor(),
                         lambda x: 2 * x - 1
                     ])):
        self.img_files = []
        for root, _, files in os.walk(root_dir):
            for file in numerical_order(files) if sorted else files:
                if UnannotatedDataset.file_is_img(file):
                    self.img_files.append(os.path.join(root, file))
        self.transform = transform

    @staticmethod
    def file_is_img(name):
        extension = os.path.basename(name).split('.')[-1]
        return extension in ['jpg', 'jpeg', 'png']

    def align_names(self, target_names):
        new_img_files = []
        img_files_names_dict = {_filename(f): f for f in self.img_files}
        for name in target_names:
            try:
                new_img_files.append(img_files_names_dict[_filename(name)])
            except KeyError:
                print('names mismatch: absent {}'.format(_filename(name)))
        self.img_files = new_img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = Image.open(self.img_files[item])
        if self.transform is not None:
            return self.transform(img)
        else:
            return img


class LabeledDatasetImagesExtractor(Dataset):
    def __init__(self, ds, img_field=0):
        self.source = ds
        self.img_field = img_field

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        return self.source[item][self.img_field]


class DatasetLabelWrapper(Dataset):
    def __init__(self, ds, label, transform=None):
        self.source = ds
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        img = self.source[item]
        if self.transform is not None:
            img = self.transform(img)
        return (img, self.label[item])


class FilteredDataset(Dataset):
    def __init__(self, source, filterer=lambda i, s: s[1], target=[], verbosity=make_verbose()):
        self.source = source
        if not isinstance(target, list):
            target = [target]
        self.indices = [i for i, s in wrap_with_tqdm(enumerate(source), verbosity)
                        if filterer(i, s) in target]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.source[self.indices[index]]


class TransformedDataset(Dataset):
    def __init__(self, source, transform, img_index=0):
        self.source = source
        self.transform = transform
        self.img_index = img_index

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source[index]
        if isinstance(out,tuple):
            return self.transform(out[self.img_index]), out[1 - self.img_index]
        else:
            return self.transform(out)


class TensorsDataset(Dataset):
    def __init__(self, source_dir):
        self.source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir)\
            if f.endswith('.pt')]

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, index):
        return torch.load(self.source_files[index]).to(torch.float32)


class RGBDataset(Dataset):
    def __init__(self, source_dataset):
        super(RGBDataset, self).__init__()
        self.source = source_dataset

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source
        if out.shape[0] == 1:
            out = out.repeat([3, 1, 1])
        return out
