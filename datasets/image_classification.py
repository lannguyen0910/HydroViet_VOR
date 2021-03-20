import random
import torch
import torch.utils.data as data
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image


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
            img_labels = sorted(os.listdir(os.path.join(self.root, cls)))
            for label in img_labels:
                data.append([f'{cls}/{label}', cls])

        if self.shuffle:
            random.shuffle(data)

        data = data[:self.n_samples] if self.n_samples is not None and self.n_samples <= len(
            data) else data

        return data

    def class_to_idx(self):
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
        """Plot distribution of classes with number"""
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
        image_name, class_name = self.data[idx]
        category = self.class_idxes[class_name]

        img_path = os.path.join(self.root, image_name)
        img = Image.open(img_path).convert('RGB')
        # width, height = img.size
        assert len(img.getbands()) == 3, 'Gray image not allow'

        if self.transforms is not None:
            img = self.transforms(img)

        return {'img': img,
                'category': category}  # cls_id

    def visualize(self, img, category, figsize=(20, 20)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)

        plt.title(self.n_classes[category])
        plt.show()

    def visualize_image(self, idx=None, figsize=(20, 20), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Visualize an image by passing idx and denormalize itself
        """
        if idx is None:
            idx = random.randint(0, len(self.data))

        item = self.__getitem__(idx)
        img = item['img']
        category = item['category']

        # denormalize to display image
        img = img.numpy().squeeze().transpose(1, 2, 0)
        img = (img * std + mean)
        img = np.clip(img, 0., 1.)

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
