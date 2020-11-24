from __future__ import division
from PIL import Image
from collections import Iterable
import torch
import random
import numpy as np
import torchvision.transforms.functional as F

# Source of torch.nn.functional
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
    :param mean: (list of float)
    :param std: (list of float)
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False, **kwargs):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, img, box=None, **kwargs):
        """
        :param img: (tensor) image to be normalized
        :param box: (list of tensor) bounding boxes to be normalized, by dividing them with image's width and heights. Format: (x,y,width,height)
        """
        new_img = F.normalize(img, mean=self.mean,
                              std=self.std, inplace=self.inplace)

        if box is not None:
            _, i_h, i_w = img.size()
            for bb in box:
                bb[0] = bb[0] * 1.0 / i_w
                bb[1] = bb[1] * 1.0 / i_h
                bb[2] = bb[2] * 1.0 / i_w
                bb[3] = bb[3] * 1.0 / i_h

        results = {
            'img': new_img,
            'box': box,
            'category': kwargs['category'],
            'mask': None
        }

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Denormalize(object):
    """
    Denormalize a tensor image and boxes to display image.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], box_transform=True, **kwargs):
        self.mean = mean
        self.std = std
        self.box_transform = box_transform

    def __call__(self, img, box=None, **kwargs):
        """
        Args:
            img_tensor (Tensor): image of size (C, H, W) to be normalized

        Return:
            old_img_numpy (Numpy)
        """
        mean = np.array(self.mean)
        std = np.array(self.std)

        img = img.numpy().squeeze().transpose(1, 2, 0)
        img = (img * std + mean)
        img = np.clip(img, 0, 1)

        if box is not None and self.box_transform:
            _, i_h, i_w = img.size()
            for bb in box:
                bb[0] = bb[0] * i_w
                bb[1] = bb[1] * i_h
                bb[2] = bb[2] * i_w
                bb[3] = bb[3] * i_h

        results = {
            'img': img,
            'box': kwargs['box'],
            'category': kwargs['category'],
            'mask': None
        }

        return results


class ToTensor(object):
    """
    Image from numpy to tensor
    """

    def __init__(self):
        pass

    def __call__(self, img, **kwargs):
        """
            :param img: (PIL Image) image to be tensorized
            :param box: (list of float) bounding boxes to be tensorized. Format: (x, y, width, height)
            :param label: (int) bounding boxes to be tensorized. Format: (x, y, width, height)
        """
        img = F.to_tensor(img)

        results = {
            'img': img,
            'box': kwargs['box'],
            'category': kwargs['category'],
            'mask': None
        }

        if kwargs['box'] is not None:
            # default: dtype = torch.float32
            box = torch.as_tensor(kwargs['box'])
            results['box'] = box

        if kwargs['category'] is not None:
            category = torch.LongTensor(kwargs['category'])
            results['category'] = category

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class Resize(object):
    """
    - Resize a PIL image with bboxes and size
    """

    def __init__(self, size=(224, 224), interpolation=Image.BILINEAR, **kwargs):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, box=None, **kwargs):
        new_img = F.resize(img, self.size, self.interpolation)

        if box is not None:
            np_box = np.array(box)
            old_dims = np.array([img.width, img.height, img.width, img.height])
            new_dims = np.array([self.size[1], self.size[0],
                                 self.size[1], self.size[0]])

            box = np.floor((np_box / old_dims) * new_dims)

        results = {
            'img': new_img,
            'box': box,
            'category': kwargs['category'],
            'mask': None
        }

        return results

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomHorizontalFlip(object):
    """
    - Horizontally Flip Image and bounding boxes + Mask
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, img, box=None, **kwargs):
        if random.random() < self.ratio:
            # Flip img
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # Flip bboxes
            if box is not None:
                img_center = np.array(np.array(img).shape[:2]) / 2
                img_center = np.hstack((img_center, img_center))

                box[:, [0, 2]] += 2*(img_center[[0, 2]] - box[:, [0, 2]])

                box_w = abs(box[:, 0] - box[:, 2])
                box[:, 0] -= box_w
                box[:, 2] += box_w

        results = {
            'img': img,
            'box': box,
            'category': kwargs['category'],
            'mask': None
        }

        return results


class Compose(object):
    """
    - Custom Transform class composes all method like Resize, ToTensor...
    """

    def __init__(self, transform_list=None):
        self.denormalize = Denormalize()

        if transform_list is None:
            self.transform_list = [Resize(), ToTensor(), Normalize()]

        else:
            self.transform_list = transform_list

        if not isinstance(self.transform_list, list):
            self.transform_list = list(self.transform_list)

    def __call__(self, img, box=None, category=None, mask=None):
        for transform in self.transform_list:
            results = transform(img=img, box=box, category=category, mask=mask)
            img = results['img']
            box = results['box']
            category = results['category']
            mask = results['mask']

        return results
