import torchvision.transforms as transforms


def transforming(img_size, style='transform_1'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if style == 'transform_1':
        normalize = transforms.Normalize(mean=mean, std=std)

        transform_train = transforms.Compose([
            transforms.Resize(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_val = transforms.Compose([
            transforms.Resize(img_size[0]),
            transforms.ToTensor(),
            normalize
        ])

        return {'train': transform_train,
                'val': transform_val,
                'test': transform_val}

    else:
        print('Wrong transform style!')
