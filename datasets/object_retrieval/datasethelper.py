import numpy as np
import cv2
import random
import torch
from skimage import transform
from skimage.util import random_noise
from PIL import Image
from configs.config import Hyperparams


def default_image_loader(path):
    return Image.open(path).convert('RGB')


def resize(img):
    wTarget, hTarget = Hyperparams.size
    h, w, _ = img.shape

    factor_x = w / wTarget
    factor_y = h / hTarget
    factor = max(factor_x, factor_y)
    newSize = (min(wTarget, int(w / factor)), min(hTarget, int(h / factor)))
    img = cv2.resize(img, newSize, cv2.INTER_NEAREST)

    target = np.zeros(shape=(hTarget, wTarget, 3), dtype=np.uint8)
    target[int((hTarget-newSize[1])/2): int((hTarget-newSize[1])/2)+newSize[1],
           int((wTarget-newSize[0])/2): int((wTarget-newSize[0])/2)+newSize[0], :] = img
    return target


def image_preprocessing(fpath):
    img = cv2.imread(fpath)
    img = resize(img)
    return img


class RandAugmentation:
    def apply(self, img):
        aug_1 = random.choice(
            [self.flip, self.rotate, self.add_noise, self.add_blur])
        aug_2 = random.choice(
            [self.flip, self.rotate, self.add_noise, self.add_blur])
        return aug_2(aug_1(img))

    def flip(self, img):
        return np.fliplr(img)

    def rotate(self, img):
        return transform.rotate(img, angle=random.choice(range(-10, 10)))

    def add_noise(self, img):
        mode = random.choice(
            ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'])
        return random_noise(img, mode=mode)

    def add_blur(self, img):
        ksize = random.choice([3, 5, 7, 9])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


def collate_fn(batch, augment=False):
    all_fpaths = [sample[0] for sample in batch]
    all_targets = [sample[1] for sample in batch]
    img_stack, target_stack = [], []
    for sub_fpaths, sub_targets in zip(all_fpaths, all_targets):
        img_tuple = []
        for fpath, _ in zip(sub_fpaths, sub_targets):
            img = image_preprocessing(fpath)
            if augment:
                img = RandAugmentation().apply(img)
            img_tuple.append(img)
        img_stack.append(img_tuple)
        target_stack.append(sub_targets)
    return torch.FloatTensor(np.transpose(img_stack, axes=(0, 1, 4, 2, 3))), torch.LongTensor(target_stack)
