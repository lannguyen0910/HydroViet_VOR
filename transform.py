import torchvision.transforms as transforms
import numpy as np


def transform(mode, img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    if mode == 'train':
        transform_train = transforms.Compose([
            transforms.Resize(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    elif mode == 'val':
        transform_val = transforms.Compose([
            transforms.Resize(img_size[0]),
            transforms.ToTensor(),
            normalize
        ])

    else:
        print('Error in transform!')

    return {'train': transform_train,
            'val': transform_val,
            'test': transform_val}


print(np.random.rand(3,))
