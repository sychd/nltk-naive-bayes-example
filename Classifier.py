from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk

import functools
import json
import codecs
import sys

from ClassifierConfiguration import ClassifierConfiguration

class Classifier:
    word_features = None
    config = None

    def __init__(self, configuration):
        self.config = configuration

    def load_data(self, filepath):
        data = None
        with codecs.open(filepath, 'r', encoding="utf-8") as json_data:
            data = json.load(json_data)
        return data

    def tokenize(self, text):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)
        return list(map(lambda x:x.lower(),tokens))

    def remove_stop_words(self, words):
        sw = stopwords.words(self.config.LANGUAGE)
        return list(filter(lambda word: word not in sw, words))

    def stem(self, words):
        stemmer = SnowballStemmer(self.config.LANGUAGE)
        return list(map(lambda word: stemmer.stem(word), words))

    def get_words(self, wordlists):
        words = list(map(lambda entry: entry[0], wordlists))
        return functools.reduce(lambda x, y: x + y, words)

    def get_frequency_distribution(self, wordlist):
        freq_dist = nltk.FreqDist(wordlist)
        bag_of_words = freq_dist.most_common()
        return list(map(lambda e: e[0], bag_of_words))

    def extract_bag_of_words(self, text):
        wordlist = self.prepare_wordlist(text)
        bag_of_words = self.extract_bag_of_words(wordlist)
        return bag_of_words

    def prepare_wordlists(self, texts, cls):
        wordlists = []
        for text in texts:
            wordlists.append(self.prepare_wordlist(text))

        return list(map(lambda word: (word, cls), wordlists))

    def prepare_wordlist(self, text):
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        tokens = self.stem(tokens)

        return tokens

    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features[word] = word in document_words
        return features

    def main(self):
        positive = self.prepare_wordlists(self.load_data(self.config.POS_FILENAME), self.config.POSITIVE_CLASS)
        negative = self.prepare_wordlists(self.load_data(self.config.NEG_FILENAME), self.config.NEGATIVE_CLASS)
        all = positive + negative

        all_words = self.get_words(all)
        self.word_features = self.get_frequency_distribution(all_words)
        training_set = nltk.classify.apply_features(self.extract_features, all)
        print(training_set)
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        print(classifier.show_most_informative_features())
        # print(self.all)
        # print(self.get_words(self.all))

        sys.exit(0)

Classifier(ClassifierConfiguration()).main()
