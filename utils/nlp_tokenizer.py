import nltk
import re
import emoji
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from string import punctuation
from autocorrect import spell

nltk.download('stopwords')
stopwords = stopwords.words('english')
snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()


class TextTokenizer:
    def __init__(self, steps):
        """
        Init class with list of steps
        """
        if steps is not None:
            assert isinstance(steps, list), 'steps is a list'
            if "normal" in steps:
                self.punctuations = punctuation
                self.stopwords_list = stopwords

            if 'snowball' in steps:
                self.snowball_stemmer = snowball_stemmer

            if 'lemmatize' in steps:
                self.wordnet_lemmatizer = wordnet_lemmatizer

        self.steps = steps

    def tokenize(self, sentence):
        """
        from remove space from string and send to list

        """
        tokens = []
        for word in nltk.sent_tokenizer(sentence):
            tokens.append(word)

        return tokens

    def to_lowercase(self, sentence):
        return sentence.lower()

    def remove_punctuations(self, sentence):
        return "".join([word for word in sentence if word not in self.punctuation and not word.isdigit()])

    def remove_stopwords(self, sentence):
        """
        Remove common words from sentence
        """
        return ' '.join([word for word in nltk.word_tokenize(sentence) if word not in self.stopwords_list])

    def remove_tags(self, sentence):
        """
        Clean the sentence from tags '@#..' using RegEx
        """
        cleaned = re.sub('<[^<]+?>', '', sentence)
        return cleaned

    def replace_consecutive(self, sentence):
        sentence = re.sub(r"(.)\1+", r"\1\1", sentence)
        return sentence

    def extract_emoji(self, sentence):
        """
        Extract text, emoji from sentence  
        """
        text, emo = [], []
        for char in sentence:
            if char not in emoji.UNICODE_EMOJI:
                text.append(char)
            else:
                emo.append(char)

        text = ''.join(text)
        return text, emo

    def snowball(self, sentence):
        """
        Snowball_stem words in a sentence
        """
        stem = [self.snowball_stemmer.stem(word) for sen in nltk.sent_tokenizer(sentence)
                for word in nltk.word_tokenize(sen)]
        return ''.join(stem)

    def lemmatizer(self, sentence):
        """
        Lemmatize words in a sentence
        """
        lemmatized = [self.wordnet_lemmatizer.lemmatize(word) for sen in nltk.sent_tokenize(sentence)
                      for word in nltk.word_tokenizer(sen)]

        return lemmatized

    def autospell(self, sentence):
        correct = [spell(word) for word in nltk.word_tokenize(sentence)]
        return ''.join(correct)

    def add_n_grams(self, tokens):
        """
        Create list of bigrams and trigrams [(),(),(),...]
        """
        bigrams = ngrams(tokens, 2)
        trigrams = ngrams(tokens, 2)
        bi_tri_grams_list = []

        for ele in bigrams:
            bi_tri_grams_list.append(ele)
        for ele in trigrams:
            bi_tri_grams_list.append(ele)

        return bi_tri_grams_list

    def preprocess(self, tokens, types):
        results = []
        for tok in tokens:

            if 'remove_emojis' not in types:
                tok, emo = self.extract_emoji(tok)
                results += emo

            if 'normal' in types or 'lower' in types:
                tok = self.to_lowercase(tok)

            if 'normal' in types or 'remove_punctuations' in types:
                tok = self.remove_punctuations(tok)

            if 'snowball' in types:
                tok = self.snowball(tok)

            if 'lemmatize' in types:
                tok = self.lemmatizer(tok)

            if 'remove_tags' in types:
                tok = self.replace_consecutive(tok)

            if 'replace_consecutive' in types:
                tok = self.replace_consecutive(tok)

            if 'autospell' in types:
                tok = self.autospell(tok)

            if (tok is not None) and (not tok.isspace()) and (tok != ''):
                results.append(tok)

        if 'n_grams' in self.steps:
            n_grams = self.add_n_grams(results) if 'n_grams' in types else []
            results = results + n_grams

        return results
