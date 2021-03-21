import os
import numpy as np
from torch.utils.data import Dataset
from .datasethelper import *


class PreProcessing(Dataset):

    images_train = np.array([])
    images_test = np.array([])
    labels_train = np.array([])
    labels_test = np.array([])
    unique_train_label = np.array([])
    map_train_label_indices = dict()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        print("Loading Geological Similarity Dataset...")
        self.images_train, self.images_test, self.labels_train, self.labels_test = self.preprocessing(
            0.8)
        self.unique_train_label = np.unique(self.labels_train)
        self.map_train_label_indices = {label: np.flatnonzero(self.labels_train == label) for label in
                                        self.unique_train_label}
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Images test  :", self.images_test.shape)
        print("Labels test  :", self.labels_test.shape)
        print("Unique label :", self.unique_train_label)

    def normalize(self, x):
        min_val = np.min(x)
        max_val = np.max(x)
        x = (x - min_val) / (max_val - min_val)
        return x

    def read_dataset(self):
        X = []
        y = []
        for directory in os.listdir(self.data_src):
            try:
                for pic in os.listdir(os.path.join(self.data_src, directory)):
                    img = image_preprocessing(
                        os.path.join(self.data_src, directory, pic))
                    X.append(np.squeeze(np.asarray(img)))
                    y.append(directory)
            except Exception as e:
                print('Failed to read images from Directory: ', directory)
                print('Exception Message: ', e)
        print('Dataset loaded successfully.')
        return X, y

    def preprocessing(self, train_test_ratio):
        X, y = self.read_dataset()
        labels = list(set(y))
        label_dict = dict(zip(labels, range(len(labels))))
        Y = np.asarray([label_dict[label] for label in y])
        # normalize images
        X = [self.normalize(x) for x in X]

        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = []
        y_shuffled = []
        for index in shuffle_indices:
            x_shuffled.append(X[index])
            y_shuffled.append(Y[index])

        size_of_dataset = len(x_shuffled)
        n_train = int(np.ceil(size_of_dataset * train_test_ratio))
        return np.asarray(x_shuffled[0:n_train]), np.asarray(x_shuffled[n_train + 1:size_of_dataset]), np.asarray(
            y_shuffled[0:n_train]), np.asarray(y_shuffled[
                                               n_train + 1:size_of_dataset])

    def get_triplets(self):
        label_l, label_r = np.random.choice(
            self.unique_train_label, 2, replace=False)
        a, p = np.random.choice(
            self.map_train_label_indices[label_l], 2, replace=False)
        n = np.random.choice(self.map_train_label_indices[label_r])
        return a, p, n

    def __getitem__(self, n):
        idxs_a, idxs_p, idxs_n = [], [], []
        for _ in range(n):
            a, p, n = self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
        return self.images_train[idxs_a, :], self.images_train[idxs_p, :], self.images_train[idxs_n, :]


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
