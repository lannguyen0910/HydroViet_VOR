from torch.utils.data import Dataset
import numpy as np
from PIL import Image

# https: // github.com/adambielski/siamese-triplet/blob/master/datasets.py


class TripletDataset(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.train = self.dataset.train
        self.transform = self.dataset.transform

        self.labels = self.dataset.labels
        self.data = self.dataset
        self.labels_set = set(self.labels.numpy())
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}

        if not self.train:
            triplets = [[i,
                         np.random.choice(
                             self.label_to_indices[self.labels[i].item()]),
                         np.random.choice(self.label_to_indices[
                             np.random.choice(
                                 list(
                                     self.labels_set - set([self.labels[i].item()]))
                             )
                         ])
                         ]
                        for i in range(len(self.data))]
            self.triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.data[index]
            label1 = label1.item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(
                    self.label_to_indices[label1])
            negative_label = np.random.choice(
                list(self.labels_set - set([label1])))
            negative_index = np.random.choice(
                self.label_to_indices[negative_label])
            img2, _ = self.data[positive_index]
            img3, _ = self.data[negative_index]
        else:
            img1, _ = self.data[self.triplets[index][0]]
            img2, _ = self.data[self.triplets[index][1]]
            img3, _ = self.data[self.triplets[index][2]]
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.dataset)
