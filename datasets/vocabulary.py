import torch.utils.data as data
import sys
from utils.nlp_tokenizer import TextTokenizer
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

sys.path.append('..')
# print(sys.path)


class VocabularyDataset(data.Dataset):
    """
    Build own vocabulary dataset of sentence from csv
    """

    def __init__(self,
                 tokenizer=None,
                 max_length=None,
                 init_token='<sos>',
                 pad_token='<pad>',
                 eos_token='<eos>',
                 unk_token='<unk>'):

        if tokenizer is None:
            self.tokenizer = TextTokenizer()
        else:
            self.tokenizer = tokenizer

        self.max_length = max_length  # Max length for vocab bag
        self.freqs = {}  # number of repeatations of a word
        self.vocab = 4

        self.special_tokens = {
            'init_token': init_token,
            'pad_token': pad_token,
            'eos_token': eos_token,
            'unk_token': unk_token
        }
        self.stoi = {
            pad_token: 0,
            init_token: 1,
            eos_token: 2,
            unk_token: 3
        }

    def build_vocab(self, data_set):
        self.dataset = data_set
        self.data = data_set.data  # data is a list of text in csv_dataset

        print('Pending ......... ')
        for sentence in tqdm(self.data):
            for token in sentence.split(''):
                if token not in self.stoi:
                    if self.max_size is not None:
                        if self.max_length <= self.vocab_size:
                            continue

                    self.stoi[token] = self.vocab_size  # index increase from 4
                    self.vocab_size += 1
                    self.freqs[token] = 1

                else:
                    self.freqs[token] += 1

        self.freqs = {tok: freq for tok, freq in sorted(
            self.freqs.items(), key=lambda item: item[1], reversed=True)}
        print('Done Bulding!')

    def most_common(self, top=None, n_grams=None):  # top-n frequencies of data
        """
        A dict of common words -> (dict)
            top: Number of top common words -> int
            n_grams: 3 grams -> str ( input = '1' or '2' or '3')

        """

        if top is None:
            top = self.max_length

        i = 0
        common_dict = {}

        if n_grams is None:
            for tok, freq in self.freqs.items():
                if i >= top:
                    break
                common_dict[tok] = freq
                i += 1
        else:
            if n_grams == '1':
                for tok, freq in self.freqs.items():
                    if i >= top:
                        break
                    if len(tok) == 1:
                        common_dict[tok] = freq
                        i += 1

            if n_grams == '2':
                for tok, freq in self.freqs.items():
                    if i >= top:
                        break
                    if len(tok) == 2:
                        common_dict[tok] = freq
                        i += 1

            if n_grams == '3':
                for tok, freq in self.freqs.items():
                    if i >= top:
                        break
                    if len(tok) == 3:
                        common_dict[tok] = freq
                        i += 1

        return common_dict

    def plotting(self, top=None, types=None, figsize=(12, 12)):
        if types is None:
            types = ['freqs', 'random']

        if 'freqs' in types:
            ax = plt.figure(figsize=figsize)

            if "random" in types:
                count_dict = self.most_common(top)
                barplot = plt.bar(list(count_dict.keys()),
                                  list(count_dict.values()), color='black')
                plt.xlabel('Unique tokens')
                plt.ylabel('Token Frequencies')
                plt.title(f'Top {top} frequencies distribution')

            else:
                if '1' in types:
                    count_dict = self.most_common(top, '1')
                    barplot = plt.bar(list(count_dict.keys()),
                                      list(count_dict.values()), color='red')
                    plt.xlabel('Unique tokens')
                    plt.ylabel('Token Frequencies')
                    plt.title(f'Top {top} frequencies distribution of 1_gram')

                if 'n_grams' in self.tokenizer.steps:
                    if '2' in types:
                        count_dict = self.most_common(top, '2')
                        barplot = plt.bar(list(count_dict.keys()),
                                          list(count_dict.values()), color='green')
                        plt.xlabel('Unique tokens')
                        plt.ylabel('Token Frequencies')
                        plt.title(
                            f'Top {top} frequencies distribution of 2_gram')

                    if '3' in types:
                        count_dict = self.most_common(top, '3')
                        barplot = plt.bar(list(count_dict.keys()),
                                          list(count_dict.values()), color='blue')
                        plt.xlabel('Unique tokens')
                        plt.ylabel('Token Frequencies')
                        plt.title(
                            f'Top {top} frequencies distribution of 3_gram')

        plt.show()
        plt.legend()

    def __len__(self) -> int:
        return self.vocab_size

    def __str__(self) -> str:
        title = 'Vocabulary Dataset\n\n'
        vocab_size = f'Number of unique words in dataset {self.vocab_size}\n'

        return title + vocab_size
