import torch.utils.data as data
import sys
from utils.nlp_tokenizer import TextTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('..')
# print(sys.path)


def convertTuple(tup):
    str = ', '.join(tup)
    return str


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

        self.init_token = init_token
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        if tokenizer is None:
            self.tokenizer = TextTokenizer(steps=['normal'])
        else:
            self.tokenizer = tokenizer

        self.max_length = max_length  # Max length for vocab bag
        self.freqs = {}  # number of repeatations of a word
        self.vocab_size = 4

        self.special_tokens = {
            'init_token': init_token,
            'eos_token': eos_token,
            'pad_token': pad_token,
            'unk_token': unk_token
        }
        self.stoi = {
            init_token: 0,
            eos_token: 1,
            pad_token: 2,
            unk_token: 3
        }

    def build_vocab(self, data_set):
        self.dataset = data_set
        self.data = data_set.text  # data is a list of text in text_dataset
        types = ['lower', 'remove_punctuations', 'snowball', 'remove_emojis'
                 'lemmatize', 'remove_tags', 'replace_consecutive', 'n_grams']
        # print('Data: ', self.data)
        print('Pending ......... ')
        for sentence in tqdm(self.data):
            # print('Sentence: ', sentence)
            for token in self.tokenizer.preprocess(sentence, types):
                # print('Token after preprocess: ', token)
                if token not in self.stoi:
                    if self.max_length is not None:
                        if self.max_length <= self.vocab_size:
                            continue

                    self.stoi[token] = self.vocab_size  # index increase from 4
                    self.vocab_size += 1
                    self.freqs[token] = 1

                else:
                    self.freqs[token] += 1
            #     break
            # break
        self.freqs = {tok: freq for tok, freq in sorted(
            self.freqs.items(), key=lambda item: item[1], reverse=True)}

        # for tok, freq in self.freqs.items():
        #     print('Tok: ', tok)

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
                    if type(tok) == str:
                        common_dict[tok] = freq
                        i += 1

            elif n_grams == '2':
                for tok, freq in self.freqs.items():
                    print('Tok 2: ', tok)
                    if i >= top:
                        break
                    if len(tok) == 2 and type(tok) == tuple:
                        common_dict[tok] = freq
                        i += 1

            elif n_grams == '3':
                for tok, freq in self.freqs.items():
                    if i >= top:
                        break
                    if len(tok) == 3 and type(tok) == tuple:
                        common_dict[tok] = freq
                        i += 1

        return common_dict

    def plotting(self, top=None, types=None, figsize=(16, 16)):
        if types is None:
            types = ['freqs', 'random']

        if 'freqs' in types:
            ax = plt.figure(figsize=figsize)

            if "random" in types:
                count_dict = self.most_common(top)
                barplot = plt.barh(list(count_dict.keys()),
                                   list(count_dict.values()), color='black')
                plt.xlabel('Unique tokens')
                plt.ylabel('Token Frequencies')
                plt.title(f'Top {top} frequencies distribution')

            else:
                if '1' in types:
                    count_dict = self.most_common(top, '1')
                    barplot = plt.barh(list(count_dict.keys()),
                                       list(count_dict.values()), color='red')
                    plt.xlabel('Unique tokens')
                    plt.ylabel('Token Frequencies')
                    plt.title(f'Top {top} frequencies distribution of 1_gram')

                if '2' in types:
                    count_dict = self.most_common(top, '2')
                    new_count_dict = {convertTuple(
                        k): v for k, v in count_dict.items()}
                    # print('Count_dict 2: ', count_dict)
                    barplot = plt.bar(list(new_count_dict.keys()),
                                      list(new_count_dict.values()), color='green')
                    plt.xlabel('Unique tokens')
                    plt.ylabel('Token Frequencies')
                    plt.title(
                        f'Top {top} frequencies distribution of 2_gram')

                if '3' in types:
                    count_dict = self.most_common(top, '3')
                    new_count_dict = {convertTuple(
                        k): v for k, v in count_dict.items()}

                    barplot = plt.bar(list(new_count_dict.keys()),
                                      list(new_count_dict.values()), color='blue')
                    plt.xlabel('Unique tokens')
                    plt.ylabel('Token Frequencies')
                    plt.title(
                        f'Top {top} frequencies distribution of 3_gram')

        plt.show()
        # plt.legend()

    def __len__(self) -> int:
        return self.vocab_size

    def __str__(self) -> str:
        title = 'Vocabulary Dataset\n\n'
        vocab_size = f'Number of unique words in dataset {self.vocab_size}\n'

        return title + vocab_size
