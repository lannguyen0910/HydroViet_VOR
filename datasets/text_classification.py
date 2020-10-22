import random
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import csv


class TextClassificationDataset(Data.Dataset):
    """
    Handling a dataset.csv with input of (category, text)
    """

    def __init__(self, root,
                 # tokenizer=str.split(''),
                 n_samples=None,
                 shuffle=False,
                 skip_header=True
                 ):
        self.root = root
        self.n_samples = n_samples
        #self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.skip_header = skip_header
        self.data = self.load_data()[0]
        self.n_categories = self.load_data()[1]

    def load_data(self):
        data = []
        with open(self.root, 'r', encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            if self.skip_header:
                next(reader)
            for line in reader:
                data.append(line)

        if self.shuffle:
            random.shuffle(data)

        data = data[:self.n_samples] if self.n_samples is not None \
            and self.n_samples <= len(data) else data

        # First Column -> Categories , Second Column -> Text
        n_categories = list(set([line[0] for line in data]))
        return (data, n_categories)

    def count_text_per_cat(self):
        count_dict = {}
        for cat, text in self.data:
            if cat in count_dict.keys():
                count_dict[cat] += 1
            else:
                count_dict[cat] = 1

        return count_dict

    def plotting(self, figsize=(12, 12), types=['freqs']):
        """Plot distribution of classes with number"""
        count_dict = self.count_text_per_cat()
        # plt.style.use('dark_background')
        ax = plt.figure(figsize=figsize)

        if 'freqs' in types:
            barplot = plt.bar(list(count_dict.keys()), list(count_dict.values()), color=[
                np.random.rand(3,) for _ in range(len(self.n_categories))])

            for rect_per_class in barplot:
                height = rect_per_class.get_height()
                plt.xlabel('Number of category')
                plt.ylabel('Number of text per category')
                plt.text(rect_per_class.get_x() + rect_per_class.get_width()/2.0,  height, '%d' %
                         int(height), ha='center', va='bottom')

            plt.title('Category Frequencies')

        plt.show()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        category, text = self.data[idx]
        tokens = text.split()
        return {'text': tokens, 'category': category}

    def __str__(self):
        title = 'Dataset for Text Classification\n\n'
        text = f'Number of text: {len(self.data)}\n'
        categories = f'Number of category: {len(self.n_categories)}\n'
        return title + text + categories
