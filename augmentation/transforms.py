from __future__ import division
from PIL import Image
from collections import Iterable
import torch
import math
import sys
import random
import numpy as np
import torchvision.transforms.functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Normalize(object):
    """
    Normalize a tensor image with default mean and standard of Imagenet deviation.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False, **kwargs):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, **kwargs):
        new_img = F.normalize(img, mean=self.mean,
                              std=self.std, inplace=self.inplace)
        return new_img, kwargs['bboxes'], kwargs['classes']


class Denormalize(object):
    """
    Denormalize a tensor image and return to numpy type.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], **kwargs):
        self.mean = mean
        self.std = std

    def __call__(self, img, bboxes, classes):
        """
        Args: 
            img_tensor (Tensor): image of size (C, H, W) to be normalized

        Return:
            old_img_numpy (Numpy)   
        """
        mean = np.array(self.mean)
        std = np.array(self.std)

        bboxes = bboxes.numpy()
        classes = classes.numpy()

        img = img.numpy().squeeze().transpose(1, 2, 0)
        img = (img * std + mean)
        img = np.clip(img, 0, 1)

        return img, bboxes, classes


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, img, bboxes,  classes, **kwargs):
        img_tensor = F.to_tensor(img)
        bboxes = torch.FloatTensor(bboxes)
        classes = torch.LongTensor(classes)

        return img_tensor, bboxes, classes

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Resize(object):
    """
    - Resize a PIL image with bboxes and size
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bboxes, **kwargs):
        new_img = F.resize(img, self.size)
        np_bboxes = np.array(bboxes)
        old_dims = np.array([img.width, img.height, img.width, img.height])
        new_dims = np.array([self.size[1], self.size[0],
                             self.size[1], self.size[0]])
        new_bboxes = np.floor((np_bboxes / old_dims) * new_dims)

        return new_img, new_bboxes, kwargs['classes']

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomHorizontalFlip(object):
    """
    Horizontally Flip Image and bounding boxes
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, img, bboxes, **kwargs):
        if random.random() < self.ratio:
            # Flip img
            img = F.hflip(img)

            # Flip bboxes
            img_center = np.array(np.array(img).shape[:2]) / 2
            img_center = np.hstack((img_center, img_center))

            bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])
            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w

        return img, bboxes, kwargs['classes']


class Compose(object):
    """
    Custom Transform class contain all method like Resize, ToTensor...
    """

    def __init__(self, transform_list=None):
        self.denormalize = Denormalize()

        if transform_list is None:
            self.transform_list = [Resize(), ToTensor(), Normalize()]

        else:
            self.transform_list = transform_list

        if not isinstance(self.transform_list, list):
            self.transform_list = list(self.transform_list)

    def __call__(self, img, bboxes, classes):
        for transform in self.transform_list:
            img, bboxes, classes = transform(
                img=img, bboxes=bboxes, classes=classes)

        return img, bboxes, classes
