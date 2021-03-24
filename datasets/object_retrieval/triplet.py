import os
import numpy as np
from torch.utils.data import Dataset
from .datasethelper import *


class TripletDataset(Dataset):
    def __init__(self, root,
                 df=None,
                 transform=None,
                 n_samples=None,
                 shuffle=False,
                 mode='train'):
        self.root = root
        self.df = df
        self.transforms = transform
        self.n_samples = n_samples
        self.mode = mode
        self.shuffle = shuffle

        if self.mode == 'train' or self.mode == 'val':
            self.n_classes = os.listdir(root)
            self.class_idxes = self.class_to_idx()
            self.data = self.load_data()
            self.index = self.df.index.values
            self.labels = self.df.iloc[:, 1].values

        else:
            self.data = self.load_test()

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

    def load_test(self):
        """Load test set"""
        data = []
        img_labels = sorted(os.listdir(self.root))
        for label in img_labels:
            data.append([f'{label}', self.root])

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

    def __getitem__(self, idx):
        anchor_name, class_name = self.data[idx]
        # category = self.class_idxes[class_name]

        anchor_path = os.path.join(self.root, anchor_name)
        anchor_img = Image.open(anchor_path).convert('RGB')

        # width, height = img.size
        assert len(anchor_img.getbands()) == 3, 'Gray image not allow'

        if self.mode == 'train' or self.mode == 'val':
            positive_list = self.index[self.index !=
                                       idx][self.labels[self.index != idx] == class_name]

            positive_idx = random.choice(positive_list)
            positive_name, _ = self.data[positive_idx]

            negative_list = self.index[self.index !=
                                       idx][self.labels[self.index != idx] != class_name]
            negative_idx = random.choice(negative_list)
            negative_name, _ = self.data[negative_idx]

            positive_path = os.path.join(self.root, positive_name)
            negative_path = os.path.join(self.root, negative_name)

            positive_img = Image.open(positive_path).convert('RGB')
            negative_img = Image.open(negative_path).convert('RGB')

            if self.transforms is not None:
                anchor_img = self.transforms(anchor_img)
                positive_img = self.transforms(positive_img)
                negative_img = self.transforms(negative_img)

            return anchor_img, positive_img, negative_img

        else:
            if self.transforms is not None:
                anchor_img = self.transforms(anchor_img)

            return anchor_img

    def __len__(self):
        return len(self.data)
