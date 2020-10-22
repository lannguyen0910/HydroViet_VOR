from datasets.vocabulary import VocabularyDataset
from datasets.text_classification import TextClassificationDataset
from utils.nlp_tokenizer import TextTokenizer

if __name__ == '__main__':
    dataset = TextClassificationDataset('bbc-text.csv')
    print(dataset)
    # dataset.plotting()

    tokenizer = TextTokenizer(
        steps=['normal', 'n_grams', 'snowball', 'lemmatize'])

    vocab = VocabularyDataset(tokenizer=tokenizer, max_length=100000)
    vocab.build_vocab(dataset)

    # print(vocab)
    # print(vocab.freqs)
    # print(vocab.most_common(6))
    # {'the': 50352, 'to': 23856, 'of': 19017, 'and': 17716, 'a': 17518, 'in': 16912}
    # vocab.plotting(top=10, types=['freqs', '1'])
    vocab.plotting(top=10, types=['freqs', '2'])
    vocab.plotting(top=10, types=['freqs', '3'])
