from builtins import print
import itertools
import nltk
import random
import pickle
from nltk.classify import ClassifierI
from statistics import mode
import collections
from nltk import BigramCollocationFinder
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.metrics import f_measure
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.metrics import BigramAssocMeasures
from analysis import analyze
from browser import Browser
from text_preprocessing import LanguageProcessing


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


class BigramsClassifier(object):
    def __init__(self):
        Browser()
        LanguageProcessing()
        self.biClassify()

    def remove_duplicates(self, numbers):
        numbers.sort()
        i = len(numbers) - 1
        while i > 0:
            if numbers[i] == numbers[i - 1]:
                numbers.pop(i)
            i -= 1
        return numbers

    def balance_pos_class(self, pos_list, neg_list):
        new_mix = list(set(pos_list).intersection(set(neg_list)))  # gets only the common words between two classes

        for item in new_mix:
            for pos_i in pos_list:
                if item in pos_list:
                    pos_list.remove(item)
                    pos_list = self.remove_duplicates(pos_list)
        return pos_list

    def balance_neg_class(self, pos_list, neg_list):
        new_mix = list(set(neg_list).intersection(set(pos_list)))  # gets only the common words between two classes

        for item in new_mix:
            for neg_i in neg_list:
                if item in neg_list:
                    neg_list.remove(item)
                    neg_list = self.remove_duplicates(neg_list)
        return neg_list

    # def find_features(self, document):
    #     words = word_tokenize(document)
    #     features = {}
    #     for w in self.word_features:
    #         features[w] = (w in words)
    #     # print("Features: ",features)
    #
    #     return features

    def bi_features(self, sentence):
        try:
            tokens = nltk.word_tokenize(sentence)
            bigram_finder = BigramCollocationFinder.from_words(tokens)
            bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 200)
            return dict([(str(ngram).lower(), True) for ngram in itertools.chain(tokens, bigrams)])
        except (AttributeError,ZeroDivisionError):
            pass

    # print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
    # bigram_fd = nltk.FreqDist(nltk.bigrams(all_words))
    # uni_featuresets = [(find_features(rev), category) for (rev, category) in documents]
    # def summaryGeneration(self,sentence):
    #
    #     fonts_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
    #     font_name = 'consolab.ttf'
    #     font = ImageFont.truetype(os.path.join(fonts_dir, font_name), 15)
    #
    #     # Opening the file gg.png
    #     imageFile = "white_correct_res.png"
    #     im1 = Image.open(imageFile)
    #
    #     # Drawing the text on the picture
    #     draw = ImageDraw.Draw(im1)
    #     draw.text((20, 10), sentence, (0, 0, 0), font=font)
    #     draw = ImageDraw.Draw(im1)
    #
    #     # Save the image with a new name
    #     im1.save("summary_text.png")
    #
    # def summaryPolarity(self,polarity,negative_rate,confidence_perc):
    #
    #     # polarity = self.finalpol
    #     #negative_rate = self.neg_perc
    #     # confidence_perc = self.confi
    #
    #     if negative_rate <= 51 and polarity == "pos":
    #         sentence="We are "+str(confidence_perc)+"% confident that mild negative words have been predicted. Just keep and eye!"
    #         self.summaryGeneration(sentence)
    #         print("1")
    #     elif negative_rate > 51 and polarity == "pos":
    #         sentence = "We are " + str(confidence_perc) + "% confident that strong negative words have been predicted. Just keep and eye!"
    #         self.summaryGeneration(sentence)
    #         print("2")
    #     elif negative_rate > 51 and polarity == "neg":
    #         sentence = "We are " + str(confidence_perc) + "% confident that very strong negative words have been predicted. Just keep and eye!"
    #         self.summaryGeneration(sentence)
    #         print("3")


    def biClassify(self):
        # short_pos = open("C:\\Users\Admin\\Dropbox\FYP\\Datasets\\New_ASB_senteced_combined\\positive4.txt", "r").read()
        # short_neg = open("C:\\Users\Admin\\Dropbox\FYP\\Datasets\\New_ASB_senteced_combined\\negative4.txt", "r").read()
        #
        # pos_documents = []
        # neg_documents = []
        # all_words = []
        # allowed_word_types = ["J", "N", "V"]  # J-Adjectives, N-nouns, V-Verb, R-Adverb
        #
        # for p in short_pos.split('\n'):
        #     pos_documents.append((p, "pos"))
        #     words = word_tokenize(p)
        #     pos = nltk.pos_tag(words)
        #     for w in pos:
        #         if w[1][0] in allowed_word_types:
        #             all_words.append(w[0].lower())
        #
        # for p in short_neg.split('\n'):
        #     neg_documents.append((p, "neg"))
        #     words = word_tokenize(p)
        #     pos = nltk.pos_tag(words)
        #     for w in pos:
        #         if w[1][0] in allowed_word_types:
        #             all_words.append(w[0].lower())
        #
        # short_pos_words = word_tokenize(short_pos)
        # short_neg_words = word_tokenize(short_neg)
        #
        # balanced_pos_words = self.balance_pos_class(short_pos_words, short_neg_words)
        # balanced_neg_words = self.balance_neg_class(short_pos_words, short_neg_words)
        #
        # for w in balanced_pos_words:
        #     all_words.append(w.lower())
        #
        # for w in balanced_neg_words:
        #     all_words.append(w.lower())

        load_all_words=open("C:\\Users\Admin\PycharmProjects\FYP_CB006302\\pickle_saves\\all_words","rb")
        all_words=pickle.load(load_all_words)
        load_all_words.close()

        all_words = nltk.FreqDist(all_words)

        # word_features = list(all_words.keys())[:5000]

        # print(word_features[:50])

        # pos_bigramfeaturesets = [(self.bi_features(rev), category) for (rev, category) in pos_documents]  # 935
        # neg_bigramfeaturesets = [(self.bi_features(rev), category) for (rev, category) in neg_documents]  # 909
        #
        # pos_featuresets = [i for i in pos_bigramfeaturesets[:909] if
        #                    i[0] != None]  # Removing None from sntences which does not have any bigrams
        # neg_featuresets = [i for i in neg_bigramfeaturesets if i[0] != None]
        # # limited_pos_bigramfeaturesets=pos_featuresets[:60]
        # # limited_neg_bigramfeaturesets=neg_featuresets[:60]
        # # bigramfeaturesets=limited_pos_bigramfeaturesets+limited_neg_bigramfeaturesets
        #
        # bigram_featuresets = pos_featuresets + neg_featuresets
        # bifeats = [(bigramExtraction.word_feats(rev), category) for (rev, category) in documents]

        # bi=BigramCollocationFinder(word_features, bigram_fd)
        # bi_featuresets=bi.score_ngrams(BigramAssocMeasures.raw_freq)
        # featuresets = uni_featuresets + bi_featuresets

        # print(len(limited_pos_bigramfeaturesets))
        # print(limited_pos_bigramfeaturesets)
        # print(len(limited_neg_bigramfeaturesets))
        # print(limited_neg_bigramfeaturesets)
        # bifeats = [(bigramExtraction.word_feats(rev), category) for (rev, category) in documents]
        # print(len(bifeats))
        # print(bifeats[:10])
        # print(bi_featuresets[:10])
        # print(len(featuresets))
        # print(len(uni_featuresets))
        # print("Featureset: ",featuresets[:10])
        # print("Bigram Featureset: ",bi_featuresets[:10])
        ###############################################################################################
        # documents = pos_documents + neg_documents
        # uni_featuresets = [(find_features(rev), category) for (rev, category) in documents]
        load_bigram_featuresets = open("C:\\Users\Admin\PycharmProjects\FYP_CB006302\\pickle_saves\\bigram_featuresets", "rb")
        bigram_featuresets = pickle.load(load_bigram_featuresets)
        load_bigram_featuresets.close()

        featuresets = bigram_featuresets
        # bi_featuresets=[(bigramExtraction.word_feats(find_features(rev)), category) for (rev, category) in documents]
        # bi_featuresets=[(bigramExtraction.word_feats(rev), category) for (rev, category) in documents]
        # featuresets=uni_featuresets+bi_featuresets
        random.shuffle(featuresets)

        # Training first 1620 features from 1942

        print(len(all_words))
        print(len(featuresets))
        # print("pos_featuresets", len(pos_featuresets))
        # print("neg_featuresets", len(neg_featuresets))

        training_set = featuresets[:1620]
        testing_set = featuresets[1620:]
        # print(len(all_words))
        print(len(featuresets))

        # classifier = nltk.NaiveBayesClassifier.train(training_set)
        # print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
        # classifier.show_most_informative_features(50)

        load_MNB_classifier = open("C:\\Users\Admin\PycharmProjects\FYP_CB006302\\pickle_saves\\MNB_classifier", "rb")
        MNB_classifier = pickle.load(load_MNB_classifier)
        load_MNB_classifier.close()

        load_SGDClassifier_classifier = open("C:\\Users\Admin\PycharmProjects\FYP_CB006302\\pickle_saves\\SGDClassifier_classifier", "rb")
        SGDClassifier_classifier = pickle.load(load_SGDClassifier_classifier)
        load_SGDClassifier_classifier.close()

        load_LinearSVC_classifier = open("C:\\Users\Admin\PycharmProjects\FYP_CB006302\\pickle_saves\\LinearSVC_classifier", "rb")
        LinearSVC_classifier = pickle.load(load_LinearSVC_classifier)
        load_LinearSVC_classifier.close()

        # MNB_classifier = SklearnClassifier(MultinomialNB())
        # MNB_classifier.train(training_set)
        # print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

        # LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        # LogisticRegression_classifier.train(training_set)
        # print("LogisticRegression_classifier accuracy percent:",
        #       (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

        # SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
        # SGDClassifier_classifier.train(training_set)
        # print("SGDClassifier_classifier accuracy percent:",
        #       (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

        ##SVC_classifier = SklearnClassifier(SVC())
        ##SVC_classifier.train(training_set)
        ##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

        # LinearSVC_classifier = SklearnClassifier(LinearSVC())
        # LinearSVC_classifier.train(training_set)
        # print("LinearSVC_classifier accuracy percent:",
        #       (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

        # NuSVC_classifier = SklearnClassifier(NuSVC())
        # NuSVC_classifier.train(training_set)
        # print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

        voted_classifier = VoteClassifier(LinearSVC_classifier, MNB_classifier, SGDClassifier_classifier)

        # src_doc = open("C:\\Users\Admin\\Dropbox\FYP\\Datasets\\CheckingPolarity\\set1_pos.txt", 'rt').readlines()
        src_doc = open("C:\\Users\Admin\PycharmProjects\FYP_CB006302\\outputfile.txt", 'rt', encoding='utf-8').readlines()
        words = []
        user_input = None
        # doc = src_doc.read().split("\n");
        for line in src_doc:
            line = line.replace("\n", "")
            line = line.replace(".", "")
            words.append(line)
            user_input = ' '.join(str(e) for e in words)
        # print(user_input)
        # print("Predicted polarity by naive bayes: ", classifier.classify(self.bi_features(user_input)))
        print("Predicted polarity by MNB_classifier: ", MNB_classifier.classify(self.bi_features(user_input)))
        # print("Predicted polarity by MNB_classifier BIGRAMS: ", MNB_classifier.classify(self.bi_features(user_input)))
        # print("Predicted polarity by LogisticRegression_classifier : ",LogisticRegression_classifier.classify(self.bi_features(user_input)))
        print("Predicted polarity by SGDClassifier_classifier: ",
              SGDClassifier_classifier.classify(self.bi_features(user_input)))
        print("Predicted polarity by LinearSVC_classifier: ",
              LinearSVC_classifier.classify(self.bi_features(user_input)))
        # print("Predicted polarity by NuSVC_classifier: ", NuSVC_classifier.classify(self.bi_features(user_input)))

        self.finalpolarity = voted_classifier.classify(self.bi_features(user_input))
        self.finalconfidence = voted_classifier.confidence(self.bi_features(user_input)) * 100

        print("Voted classifier accuracy (%):", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)
        print("Classification:", self.finalpolarity,
              "Confidence %:", self.finalconfidence)

        # Precison and recall calculations
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (feats, label) in enumerate(testing_set):
            refsets[label].add(i)
            observed = voted_classifier.classify(feats)
            testsets[observed].add(i)

        # Higher the precision lesser the false positive
        print('Positive precision:', precision(refsets['pos'], testsets['pos']))
        # Higher the recall lesser the false negative
        print('Positive recall:', recall(refsets['pos'], testsets['pos']))
        # Higher the better
        print('Positive F-measure:', f_measure(refsets['pos'], testsets['pos']))

        print('Negative precision:', precision(refsets['neg'], testsets['neg']))
        print('Negative recall:', recall(refsets['neg'], testsets['neg']))
        print('Negative F-measure:', f_measure(refsets['neg'], testsets['neg']))

        # Analyzing sentences to get polarity of each sentence
        sa_words = []
        for line in user_input.split('\n'):
            sa_words.append(line)

        sid = SentimentIntensityAnalyzer()
        res_dic = {"Positive": 0, "Negative": 0}
        for sentence in sa_words:
            sentiment_score = sid.polarity_scores(sentence)
            if sentiment_score["compound"] < 0.0:
                res_dic["Negative"] += 1
            elif sentiment_score["compound"] > 0.0:
                res_dic["Positive"] += 1

        print(res_dic)
        print("Compound value: ", sentiment_score["compound"])

        # Classifying sentences
        results = {"Positive": 0, "Negative": 0}
        number_sentences = 0
        negative_sentences = []
        for sentence in src_doc:
            number_sentences += 1
            line = sentence.replace("\n", "").replace(".", "")
            polarity = voted_classifier.classify(self.bi_features(line))
            if str(polarity) == 'pos':
                results["Positive"] += 1
            elif str(polarity) == 'neg':
                negative_sentences.append(line)
                results["Negative"] += 1

        polarity_pos_percentage = (results["Positive"] / number_sentences)
        polarity_neg_percentage = (results["Negative"] / number_sentences)
        print("Positive percentage: ", polarity_pos_percentage * 100)
        print("Negative percentage: ", polarity_neg_percentage * 100)


        #self.summaryPolarity(self.finalpolarity,polarity_neg_percentage,self.finalconfidence)



        # plot_piechart(polarity_pos_percentage,polarity_neg_percentage)
        # instanceSummary = summary()
        # instanceSummary.sendSummary(polarity_neg_percentage)
        # instanceSummary.sendSummary(polarity_neg_percentage)
        # instanceAnalyze.summaryPolarity(self.finalpolarity, polarity_neg_percentage, self.finalconfidence)
        instanceAnalyze = analyze(self.bi_features, self.finalpolarity, polarity_neg_percentage*100, self.finalconfidence)

        instanceAnalyze.plot_piechart(polarity_pos_percentage * 100, polarity_neg_percentage * 100)
        instanceAnalyze.createWordCloudfile(negative_sentences)

        instanceAnalyze.summaryPolarity(self.finalpolarity,polarity_neg_percentage,self.finalconfidence)

        # if negative_rate<= 51 and  self.finalpolarity=="pos":
        #     #MyForm.summaryText("mild",self.finalconfidence)
        #     print("1")
        # elif negative_rate> 51 and polarity=="pos":
        #    # MyForm.summaryText("strong", self.finalconfidence)
        #     print("2")
        # elif negative_rate> 51 and polarity=="neg":
        #     #MyForm.summaryText("very strong", self.finalconfidence)
        #     print("3")

# if __name__ == "__main__":
#         BigramsClassifier()
