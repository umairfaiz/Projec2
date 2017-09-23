import itertools
from functools import partial
from operator import is_not

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
import numpy


def get_bigrams(myString):
    #myString=['this is a sentence', 'so is this one', 'cant is a railway station','citadel hotel',' police stn']
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(myString)
    #stemmer = PorterStemmer()
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
    #print("bigrams", bigrams)
    for bigram_tuple in bigrams:
        x = "(%s, %s)" % bigram_tuple
        tokens.append(x)

    result = [' '.join([w.lower() for w in x.split()]) for x in tokens if
              x.lower()and len(x)>1]
    #print("result ",result)
    return result

def word_feats(words):

    bigram_score = BigramAssocMeasures.chi_sq
    tokens = nltk.word_tokenize(words)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    # bigram_finder.apply_freq_filter(3)
    all_bigrams = bigram_finder.nbest(bigram_score, 80)
    contain_word = ['not', 'no']

    bigrams = [w for w in all_bigrams if (w[0] in contain_word)]  # picking only bigrams which starts with no, not
    #return bigrams
    #return [(ngram, True) for ngram in itertools.chain(words, bigrams) if len(ngram) > 1]
    if([(ngram, True) for ngram in itertools.chain(words, bigrams) if len(ngram) > 1]!=[]):

        return dict([(str(ngram).lower(), True) for ngram in itertools.chain(words, bigrams) if len(ngram) > 1])

        # return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)if len(ngram)>1])

def newBigram(words):
    bigram_score = BigramAssocMeasures.chi_sq
    n = 200
    tokens = nltk.word_tokenize(words)
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(bigram_score, n)
    return dict([(str(ngram).lower(), True) for ngram in itertools.chain(tokens, bigrams)])

    # bigram_score = BigramAssocMeasures.chi_sq
    # tokens = nltk.word_tokenize(words)
    # bigram_finder = BigramCollocationFinder.from_words(tokens)
    # # bigram_finder.apply_freq_filter(3)
    # all_bigrams = bigram_finder.nbest(bigram_score, 500)
    #
    # return dict([(str(ngram).lower(), True) for ngram in itertools.chain(words, all_bigrams)])


def bigramf(words):
    text = 'life dream come t'
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    tokens = nltk.wordpunct_tokenize(words)
    finder = BigramCollocationFinder.from_words(tokens)
    scored = finder.score_ngrams(bigram_measures.raw_freq)
    return sorted(bigram for bigram, score in scored)
    # print(s)


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


def bigramTF(words):
    lectures = ['this is some food', 'this is some drink']
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(lectures)
    features_by_gram = defaultdict(list)
    for f, w in zip(vectorizer.get_feature_names(), vectorizer.idf_):
        features_by_gram[len(f.split(' '))].append((f, w))
    top_n = 2
    for gram, features in features_by_gram.items():
        top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [f[0] for f in top_features]
        print('{}-gram top:'.format(gram), top_features)


def bigramPrediction(words, labels):
    countVectorizer = CountVectorizer(stop_words=None, encoding='utf-8', ngram_range=(2, 2))

    fit_transform = countVectorizer.fit_transform(words)
    vocab = countVectorizer.vocabulary_

    labels = numpy.asarray(labels)

    SGDclassifier = SGDClassifier(loss='hinge', penalty='l1')
    SGDclassifier.fit(fit_transform, labels)

    countVectorizer = CountVectorizer(vocabulary=vocab, ngram_range=(2, 2))

    return countVectorizer


if __name__ == "__main__":

    sentence = ['no this is a sentence', 'so is this one is a sentence', 'cant is railway station', 'citadel hotel',
                ' police stn', 'I do not like green eggs and ham, I do not like them Sam I am!, no ham']
    # sentence ='life', 'dream', 'come', 't', 'starts'
    # word_feats(sentence)

    for line in sentence:
        features = newBigram(line)

        print(features)
