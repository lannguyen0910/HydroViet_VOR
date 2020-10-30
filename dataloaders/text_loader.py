import torch


class MyTextCollator:
    """
    - My_collate function for loading text to data_loader
    - Args:
            + dataset:                  text classification dataset (class)
            + vocab:                    Vocab dataset built from dataset (class)
            + sort_within_batch         if sort batch of (text, category, length) by length (bool)
            + batch_first               reshape output tensors to have batch_size in first index (bool)
            + include_lengths           add length to each  item (text, category) (bool)
            + device                    use CUDA or whatever (None)
    """

    def __init__(self, dataset, vocab, sort_within_batch=False, batch_first=False, include_lengths=False, device=None):
        self.dataset = dataset
        self.vocab = vocab  # Vocabulary dataset
        self.device = device
        self.batch_first = batch_first
        self.include_lengths = include_lengths
        self.sort_within_batch = sort_within_batch

    def toks_to_idxs(self, data_sample, max_length):
        """
        Take the text (list of word) in a sentence (from text_classification) and convert it to idx -> (dict)
        """
        init_idx = self.vocab.stoi[self.vocab.init_token]
        pad_idx = self.vocab.stoi[self.vocab.pad_token]
        eos_idx = self.vocab.stoi[self.vocab.eos_token]

        # Convert token to index
        tokens, categories = data_sample['text'], data_sample['category']

        idx_list = [init_idx]
        for tok in tokens:
            idx_list.append(self.vocab.stoi[tok])

        idx_list.append(eos_idx)

        length = len(idx_list)

        while len(idx_list) < max_length:
            idx_list.append(pad_idx)

        target = self.dataset.category_idx[categories]

        results = {
            'text': idx_list,
            'category': target
        }

        if self.include_lengths:
            results['lengths'] = length

        return results

    def sort_batch(self, idx_batch):  # batch has been converted to idx_batch in __call__
        """
        Sort batch by length in desc order -> (list)
        """
        assert self.include_lengths, 'include lengths - sort by lengths'
        data_list = [data['text'] for data in idx_batch]
        label_list = [data['category'] for data in idx_batch]
        length = [data['lengths'] for data in idx_batch]

        sort_idx_batch = [[a, b, c] for a, b, c in sorted(
            zip(length, data_list, label_list), reverse=True)]

        new_idx_batch = []
        for text, category, lengths in sort_idx_batch:
            new_idx_batch.append({'text': text,
                                  'lengths': lengths,
                                  'category': category})

        return new_idx_batch

    def __call__(self, batch):
        """
        Make batch, like batch_size in dataloader -> (dict)
        """
        max_length = 0
        for data in batch:
            max_length = max(len(data['text']), max_length)

        max_length += 2

        batch_idx = []
        for data in batch:
            idx_list = self.toks_to_idxs(data, max_length)
            batch_idx.append(idx_list)

        if self.sort_within_batch:
            batch_idx = self.sort_batch(batch_idx)

        data_list = [data['text'] for data in batch]
        label_list = [data['category'] for data in batch]

        if self.include_lengths:
            length = [data['lengths'] for data in batch]
            length = torch.LongTensor(length)
            length = length.to(self.device) if self.device else length

        data_list = torch.LongTensor(data_list)

        if self.batch_first:
            data_list = data_list.permute(1, 0)

        label_list = torch.LongTensor(label_list)

        if self.device is not None:
            data_list = data_list.to(self.device)
            label_list = label_list.to(self.device)

        results = {
            'text': data_list,
            'category': label_list
        }

        if self.include_lengths:
            results['lengths'] = length  # it's okay to use

        return results
