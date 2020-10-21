import torch.utils.data as data
import torch
import os
import sys
sys.path.append('..')
print(sys.path)


class VocabularyDataset(data.Dataset):
    def __init__(self,
                 max_length=None,
                 init_token='<sos>',
                 pad_token='<pad>',
                 eos_token='<eos>',
                 unk_token='<unk>'):
        self.max_length = max_length

        self.vocab = 4
        self.special_tokens = {
            'init_token': init_token,
            'pad_token': pad_token,
            'eos_token': eos_token,
            'unk_token': unk_token
        }
        self.stoi = {
            'init_token': 0,
            'pad_token': 1,
            'eos_token': 2,
            'unk_token': 3
        }
