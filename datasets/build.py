# Copyright (c). All Rights Reserved

"""
dataset which returns image and target for evaluation.
target contains the file name and image label of the image.
"""
import os
import torch
from pathlib import Path
import datasets.transforms as T
from torchvision.datasets import ImageFolder

from torchvision import transforms
from .constants import data_mean_std
from timm.data import create_transform

from PIL import Image
from transformers import ViTImageProcessor

class CreateImageFolder(ImageFolder):
    def __init__(self, root, transform=None, args=None):
        super().__init__(root, transform=transform)
        self.args = args

    def __getitem__(self, index):
        filename, imagelabel = self.samples[index]
        img = self.loader(filename)
        target={}
        target["file_name"]= [filename]
        target["image_label"]=torch.tensor([imagelabel], dtype=torch.int64) 
        
        if self.transform is not None:
            if self.args.data_transform=="swin":
                img = self.transform(img)
            if self.args.data_transform=="vit":
                    image = Image.open(filename)
                    image_rgb = image.convert('RGB')
                    img_transform=self.transform(image_rgb, return_tensors="pt")
                    img=img_transform.pixel_values.squeeze(0)
            if self.args.data_transform=="detr":
                img, target = self.transform(img, target)

        return img, target


# The transform follows DETR transform 
def detr_transforms(image_set, args):

    if args.dataset_name in data_mean_std:
        mean, std = data_mean_std[args.dataset_name]
    else:
        raise RuntimeError(
            f"Can't find mean/std for {args.dataset_name}. Please add it to dataset/constants.py"
        )

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

try:
    from torchvision.transforms import InterpolationMode
    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR

    import timm.data.transforms as timm_transforms
    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

# transformation used in swin transformer
def swin_transform(is_train, args):

    if args.dataset_name in data_mean_std:
        mean, std = data_mean_std[args.dataset_name]
    else:
        raise RuntimeError(
            f"Can't find mean/std for {args.dataset_name}. Please add it to dataset/constants.py"
        )

    resize_im = args.img_size > 32
    if is_train:
        transform = create_transform(
            input_size= args.img_size,
            is_training=True,
            color_jitter=args.AUG_COLOR_JITTER if args.AUG_COLOR_JITTER > 0 else None,
            auto_augment=args.AUG_AUTO_AUGMENT if args.AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=args.AUG_REPROB, 
            re_mode=args.AUG_REMODE,
            re_count=args.AUG_RECOUNT,
            interpolation= args.DATA_INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.img_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.TEST_CROP:
            size = int((256 / args.img_size) * args.img_size)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp('bicubic')),
            )
            t.append(transforms.CenterCrop(args.img_size))
        else:
            t.append(
                transforms.Resize((args.img_size, args.img_size),
                                  interpolation=_pil_interp('bicubic'))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# transformation used in ViT transformer 
# (https://github.com/lucidrains/vit-pytorch/blob/main/examples/cats_and_dogs.ipynb)
def vit_transform(is_train, args):

    if args.dataset_name in data_mean_std:
        mean, std = data_mean_std[args.dataset_name]
    else:
        raise RuntimeError(
            f"Can't find mean/std for {args.dataset_name}. Please add it to dataset/constants.py"
        )

    if is_train:
        transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    else:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transform


def vit_transform_1(is_train, args):

    if args.dataset_name in data_mean_std:
        mean, std = data_mean_std[args.dataset_name]
    else:
        raise RuntimeError(
            f"Can't find mean/std for {args.dataset_name}. Please add it to dataset/constants.py"
        )

    resize_im = args.img_size > 32

    if is_train:
        transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    else:
        transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomResizedCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transform


def build(image_set, args):

    root = Path(args.dataset_path)
    assert root.exists(), f'provided dataset path {root} does not exist'
    data_file = Path(args.dataset_name)

    PATHS = {
        "train": (root / data_file / "train"),
        "val": (root / data_file / "val"),
        "test": (root / data_file / "test"),
    }

    img_folder = PATHS[image_set]

    if args.data_transform=='detr':
        if image_set == 'train':
            transform = detr_transforms(image_set, args)
        elif image_set == 'val' or image_set == 'test':
            transform = detr_transforms(image_set, args)
        else:
            raise ValueError(f'unknown {image_set}')
    if args.data_transform=='swin':
        if image_set == 'train':
            transform = swin_transform(True, args)
        elif image_set == 'val' or image_set == 'test':
            transform = swin_transform(False, args)
        else:
            raise ValueError(f'unknown {image_set}')
    if args.data_transform=='vit':
        if image_set == 'train':
            transform = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        elif image_set == 'val' or image_set == 'test':
            transform = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    dataset = CreateImageFolder(root=img_folder, transform=transform, args=args)

    return dataset

