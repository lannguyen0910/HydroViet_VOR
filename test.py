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
    print(vocab)
    print(vocab.most_common(10))
    # vocab.plotting(top=10, types=['freqs', '1'])
