import random
import torch.utils.data as Data
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image


class ImageClassificationDataset(Data.Dataset):
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
        count_dict = self.count_img_per_class()
        # plt.style.use('dark_background')
        ax = plt.figure(figsize=figsize)

        if 'freqs' in types:
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
        label, cls = self.data[idx]
        #cls_id = self.class_idxes[cls]

        img_path = os.path.join(self.root, label)
        img = Image.open(img_path).convert('RGB')
        assert len(img.getbands()) == 3, 'Gray image or sth'

        if self.transforms:
            root_name = os.path.dirname(self.root)
            if self.mode == 'train':
                img = self.transforms(img)['train']
            elif self.mode == 'val':
                img = self.transforms(img)['val']
            elif self.mode == 'test':
                img = self.transforms(img)['test']
            else:
                print('Error! Please rename your folder')

        return {'img': img,
                'label': cls}  # cls_id

    def __str__(self):
        title = 'Dataset for Image Classification\n\n'
        samples = f'Number of samples: {len(self.data)}\n'
        classes = f'Number of classes: {len(self.n_classes)}\n'
        return title + samples + classes
