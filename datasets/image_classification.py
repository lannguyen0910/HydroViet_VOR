import random
import torch
import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from augmentation.transforms import Denormalize
import albumentations as A


class ImageClassificationDataset(data.Dataset):
    def __init__(self, root,
                 transforms=None,
                 n_samples=None,
                 shuffle=False,
                 mode='train'):

        self.root = root
        self.transforms = transforms
        self.n_samples = n_samples
        self.mode = mode
        self.shuffle = shuffle
        self.n_classes = os.listdir(root)
        self.class_idxes = self.class_to_idx()
        self.data = self.load_data()

    def load_data(self):
        """Load images and shuffle it -> data (list)"""
        data = []

        for cls in self.n_classes:
            data_labels = sorted(os.listdir(os.path.join(self.root, cls)))
            for img_name in data_labels:
                img_path = os.path.join(self.root, cls, img_name)
                data.append([img_path, cls])

        if self.shuffle:
            random.shuffle(data)

        data = data[:self.n_samples] if self.n_samples is not None and self.n_samples <= len(
            data) else data

        return data

    def class_to_idx(self):
        """
        Convert class_name -> class_idxes
        """
        class_idxes = {}
        idx = 0
        for cls in self.n_classes:
            class_idxes[cls] = idx
            idx += 1

        return class_idxes

    def count_img_per_class(self):
        """Count how many images in one class -> count_dict (dict)"""
        count_dict = {}
        for cls in self.n_classes:
            count_imgs = len(os.listdir(os.path.join(self.root, cls)))
            count_dict[cls] = count_imgs

        return count_dict

    def plotting(self, figsize=(12, 12), types=['freqs']):
        """
        Plot distribution of classes with number
        Call after init the image classification dataset
        """
        # plt.style.use('dark_background')
        ax = plt.figure(figsize=figsize)

        if 'freqs' in types:
            count_dict = self.count_img_per_class()
            barplot = plt.bar(list(count_dict.keys()), list(count_dict.values()), color=[
                np.random.rand(3,) for _ in range(len(self.n_classes))])

            for rect_per_class in barplot:
                height = rect_per_class.get_height()
                plt.xlabel('Number of class')
                plt.ylabel('Number of image per class')
                plt.text(rect_per_class.get_x() + rect_per_class.get_width()/2.0, height, '%d' %
                         int(height), ha='center', va='bottom')

            plt.title('Classes Frequencies')

        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        category = self.class_idxes[class_name]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        # This is for PIL format
        # assert len(img.getbands()) == 3, 'Gray image not allow'

        if self.transforms is not None:
            img = self.transforms(image=img)

        category = torch.LongTensor([category])
        return {'img': img,
                'category': category}  # cls_id

    def visualize(self, img, category, figsize=(20, 20)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

        plt.title(self.n_classes[category])
        plt.show()

    def visualize_image(self, idx=None, figsize=(20, 20)):
        """
        Visualize an image by passing idx and denormalize itself
        """
        if idx is None:
            idx = random.randint(0, len(self.data))

        item = self.__getitem__(idx)
        img = item['img']
        category = item['category']

        # Denormalize and reverse-tensorize
        normalize = False
        if self.transforms is not None:
            for x in self.transforms.transforms:
                if isinstance(x, A.Normalize):
                    normalize = True
                    denormalize = Denormalize(mean=x.mean, std=x.std)

        # Denormalize and reverse-tensorize
        if normalize:
            img = denormalize(img=img['image'])

        category = category.numpy().item()
        self.visualize(img, category, figsize=figsize)

    def collate_fn(self, batch):
        images = torch.stack([b['img'] for b in batch], dim=0)
        categories = torch.Tensor([b['category']
                                   for b in batch]).to(dtype=torch.LongTensor)

        return {
            'imgs': images,
            'categories': categories
        }

    def __str__(self):
        title = 'Dataset for Image Classification\n\n'
        samples = f'Number of samples: {len(self.data)}\n'
        classes = f'Number of classes: {len(self.n_classes)}\n'
        return title + samples + classes
