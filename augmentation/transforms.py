import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

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


transforms_train = A.Compose([
    A.SmallestMaxSize(max_size=256),
    A.RandomCrop(width=224, height=224),

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
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
