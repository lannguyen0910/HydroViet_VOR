import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

# standards for ImageNet dataset
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Denormalize:
    def __init__(self, mean=MEAN, std=STD):
        self.mean = mean
        self.std = std

    def __call__(self, img, box=None, label=None, mask=None, **kwargs):
        mean = np.array(self.mean)
        std = np.array(self.std)

        img_ = img.numpy().squeeze().transpose((1, 2, 0))
        img_ = (img_ * std) + mean
        img_ = np.clip(img_, 0, 1)

        return img_


# albumentations-teams example notebook
def augment_flips_color(p=.5):
    return A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                           rotate_limit=45, p=.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue()
    ], p=p)


def strong_aug(p=.5):
    return A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),

        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),

        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=.1),
            A.Blur(blur_limit=3, p=.1),
        ], p=0.2),

        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2,
                           rotate_limit=45, p=.2),

        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1),
            A.IAAPiecewiseAffine(p=0.3),
        ], p=0.2),

        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomContrast(),
            A.RandomBrightness(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=p)


def get_augmentation(config, type='train'):
    transforms_train = A.Compose([
        A.Resize(
            height=config.image_size[1],
            width=config.image_size[0]),

        A.OneOf([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15,
                               border_mode=cv2.BORDER_CONSTANT, value=0),
            A.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0),
            A.NoOp(),
        ]),
        A.OneOf([
            A.RandomGamma(gamma_limit=(50, 150)),
            A.NoOp()
        ]),

        A.OneOf([
            A.Blur(),
            A.Transpose(),
            A.ElasticTransform(),
            A.GridDistortion(),
            A.CoarseDropout(),
            A.NoOp()
        ]),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.NoOp()
        ]),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transforms_val = A.Compose(
        [
            A.Resize(
                height=config.image_size[1],
                width=config.image_size[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    return transforms_train if type == 'train' else transforms_val
