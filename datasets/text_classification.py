import random
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import csv


class TextClassificationDataset(Data.Dataset):
    """
    - Handling a dataset.csv with input of (category, text)
    - Args:
            + root:                 path to csv file (str)
            + tokenizer             preprocess text (default string split)
            + n_samples             number of text to use (n_samples must <= max_samples)
            + shuffle               shuffle text for randomness
            + skip_header           remove header of csv file
    """

    def __init__(self, root,
                 train,
                 tokenizer=str.split(''),
                 n_samples=None,
                 shuffle=False,
                 skip_header=True
                 ):
        super.__init__()
        self.root = root
        self.train = train
        self.n_samples = n_samples
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.skip_header = skip_header
        self.data = self.load_data()[0]
        self.n_categories = self.load_data()[1]
        self.text = self.load_data()[2]

        self.category_idx = self.category_to_idx()
        self.idx_category = {v: k for k, v in self.category_idx.items()}

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
        text = list(set([text[1] for text in data]))
        return (data, n_categories, text)

    def category_to_idx(self):
        """
        - From category to index
        """
        idx_dict = {}
        idx = 0
        for cat in self.n_categories:
            idx_dict[cat] = idx
            idx += 1

        return idx_dict

    def count_text_per_cat(self):
        """
        - How many texts has same category
        """
        count_dict = {}
        for cat, text in self.data:
            if cat in count_dict.keys():
                count_dict[cat] += 1
            else:
                count_dict[cat] = 1

        return count_dict

    def plotting(self, figsize=(12, 12), types=['freqs']):
        """
        Visualize distribution of classes with frequencies
        """
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
        tokens = self.tokenizer.tokenize(text)
        return {'text': tokens, 'category': category}

    def __str__(self):
        title = 'Dataset for Text Classification\n\n'
        text = f'Number of unique text: {len(self.text)}\n'
        categories = f'Number of unique category: {len(self.n_categories)}\n'

        return title + text + categories
