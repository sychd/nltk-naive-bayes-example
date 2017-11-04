from ClassifierConfiguration import ClassifierConfiguration
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import nltk
import functools
import json
import codecs
import sys

class Classifier:
    classifier = None
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

    def get_words_in_freq_sequence(self, wordlist):
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

    # Returns dict of words that were encountered both in train set
    # and target document with True, otherwise with False value
    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features[word] = word in document_words
        return features

    def train(self, documents):
        self.word_features = self.get_words_in_freq_sequence(self.get_words(documents))
        training_set = nltk.classify.apply_features(self.extract_features, documents)
        self.classifier = nltk.NaiveBayesClassifier.train(training_set)

        self.classifier.show_most_informative_features()
        print("\n")
        return None

    def test(self, sentiment):
        test = self.load_data(self.config.TEST_FILENAME)[sentiment];

        for document in test:
            wordlist = self.prepare_wordlist(document)
            features = self.extract_features(wordlist)
            result = self.classifier.classify(features)

            preview_text_len = self.config.PREVIEW_TEXT_LENGTH
            preview_text = document if len(document) < preview_text_len else document[:preview_text_len] + "..."
            print("%s -> %s (%r)" % (preview_text, result, result == sentiment))

        return None

    def main(self):
        pos_documents = self.prepare_wordlists(self.load_data(self.config.POS_FILENAME), self.config.POSITIVE_CLASS)
        neg_documents = self.prepare_wordlists(self.load_data(self.config.NEG_FILENAME), self.config.NEGATIVE_CLASS)
        all_documents = pos_documents + neg_documents

        self.train(all_documents)
        self.test(self.config.POSITIVE_CLASS)
        self.test(self.config.NEGATIVE_CLASS)

        sys.exit(0)

Classifier(ClassifierConfiguration()).main()
