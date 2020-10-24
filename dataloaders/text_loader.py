import torch


class MyTextCollator:
    def __init__(self, dataset, vocab, device=None):
        self.dataset = dataset
        self.vocab = vocab  # Vocabulary dataset
        self.device = device

    def toks_to_idxs(self, data_sample, max_length):
        """
        Take the text (list of word) in a sentence (from text_classification) and convert it to idx 
        """
        init_idx = self.vocab.stoi[self.vocab.init_token]
        pad_idx = self.vocab.stoi[self.vocab.pad_token]
        eos_idx = self.vocab.stoi[self.vocab.eos_token]

        tokens, categories = data_sample['text'], data_sample['category']

        idx_list = [init_idx]
        for tok in tokens:
            idx_list.append(self.vocab.stoi[tok])

        idx_list.append(eos_idx)

        length = len(idx_list)

        while len(idx_list) < max_length:
            idx_list.append(pad_idx)

        target = self.dataset.n_classes

        return {
            'text': idx_list,
            'category': target
        }

    def __call__(self, batch):
        max_length = 0
        for data in batch:
            max_length = max(len(data['text']), max_length)

        max_length += 2

        batch_idx = []
        for data in batch:
            idx_list = self.toks_to_idxs(data, max_length)
            batch_idx.append(idx_list)

        data_list = [data['text'] for data in batch]
        label_list = [data['category'] for data in batch]

        data_list = torch.LongTensor(data_list)
        label_list = torch.LongTensor(label_list)

        if self.device is not None:
            data_list = data_list.to(self.device)
            label_list = label_list.to(self.device)

        return data_list, label_list
