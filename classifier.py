import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize, sent_tokenize

import collections
from nltk import FreqDist, BigramAssocMeasures
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.metrics import f_measure
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open("C:\\Users\Admin\\Dropbox\FYP\\Datasets\\New_ASB_senteced_combined\\positive.txt", "r").read()
short_neg = open("C:\\Users\Admin\\Dropbox\FYP\\Datasets\\New_ASB_senteced_combined\\negative.txt", "r").read()

documents = []
all_words = []
allowed_word_types = ["J", "N", "V"]  # J-Adjectives, N-nouns, V-Verb, R-Adverb

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)

# positive data example:
training_set = featuresets[:100]
testing_set = featuresets[100:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

# BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
# BernoulliNB_classifier.train(training_set)
# print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

voted_classifier = VoteClassifier(
    LinearSVC_classifier,
    MNB_classifier,
    SGDClassifier_classifier)

# user_input = "great film. 10/10"
# user_input = "would like recommend matinee anybody good humorous story"
# user_input ="Suicide, kill murder terrorism drugs self injury"

#src_doc = open("C:\\Users\Admin\\Dropbox\FYP\\Datasets\\CheckingPolarity\\set1_pos.txt", 'rt').readlines()
src_doc = open("NewchromehistoryPOS_log.txt", 'rt',encoding='utf-8').readlines()
words = []
# doc = src_doc.read().split("\n");
for line in src_doc:
    line = line.replace("\n", "")
    line = line.replace(".", "")
    words.append(line)
    user_input = ' '.join(str(e) for e in words)
    # user_input= word_tokenize(str1)
# print(user_input)

print("Predicted polarity by naive bayes: ", classifier.classify(find_features(user_input)))
print("Predicted polarity by MNB_classifier: ", MNB_classifier.classify(find_features(user_input)))
print("Predicted polarity by LogisticRegression_classifier : ",
      LogisticRegression_classifier.classify(find_features(user_input)))
print("Predicted polarity by SGDClassifier_classifier: ", SGDClassifier_classifier.classify(find_features(user_input)))
print("Predicted polarity by LinearSVC_classifier: ", LinearSVC_classifier.classify(find_features(user_input)))
print("Predicted polarity by NuSVC_classifier: ", NuSVC_classifier.classify(find_features(user_input)))

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
print("Classification:", voted_classifier.classify(find_features(user_input)),
      "Confidence %:", voted_classifier.confidence(find_features(user_input)) * 100)

# Precison and recall calculation
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)

for i, (feats, label) in enumerate(testing_set):
    refsets[label].add(i)
    observed = classifier.classify(feats)
    testsets[observed].add(i)

# 35% false positives for the pos label.
print('Positive precision:', precision(refsets['pos'], testsets['pos']))
# 98% recall, so very few false negatives
print('Positive recall:', recall(refsets['pos'], testsets['pos']))
# higher the better
print('Positive F-measure:', f_measure(refsets['pos'], testsets['pos']))

print('Negative precision:', precision(refsets['neg'], testsets['neg']))
print('Negative recall:', recall(refsets['neg'], testsets['neg']))
print('Negative F-measure:', f_measure(refsets['neg'], testsets['neg']))

sa_words = []
for line in user_input.split('\n'):
    sa_words.append(line)

sid = SentimentIntensityAnalyzer()
res = {"Positive": 0, "Negative": 0}
for sentence in sa_words:
    ss = sid.polarity_scores(sentence)
    if ss["compound"] < 0.0:
        res["Negative"] += 1
    elif ss["compound"] > 0.0:
        res["Positive"] += 1
        # else:
        #     res["less"] +=1
print(res)