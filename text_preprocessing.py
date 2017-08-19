import re
import enchant
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag


class LanguageProcessing(object):
    def __init__(self):
        pass

    def tokeniz(self, file_name):

        # filename = open("history1.csv", "r")
        tokens = []
        for line in file_name.readlines():
            #line = line.decode('ASCII', 'ignore')
            # line = line.lower()  # converting words to lower case
            tokens += word_tokenize(line)
        return tokens
        # self.clean_URL(tokens)
        # filename.close()

    def clean_URL(self, f):  # need to call this method in tokenize
        cleanedurl = []
        for line in f:
            line = line.lower()
            #line = line.decode('utf-8', 'ignore')
            line = line.replace('+', ' ').replace('.', ' ').replace('=', ' ').replace('-', ' ').replace('/',
                                                                                                        ' ').replace(
                '//', ' ').replace(',', ' ').replace('www', ' ').replace('https', ' ').replace('http', ' ').replace(':', ' ').replace('\'', ' ')
            line = re.sub("(^|\W)\d+($|\W)", '', line)
            line = line.strip(' ')
            # self.remove_stopwords(line)
            cleanedurl.append(line)
        new_line = filter(lambda x: len(x) > 1, cleanedurl)  # removing the spaces from text
        # return cleanedurl
        return new_line
        # self.remove_non_englishwords(new_line)

    def remove_non_englishwords(self, w):
        d = enchant.Dict("en_US")
        string = " ".join(w)
        english_words = []
        for word in string.split():
            if d.check(word):
                if len(word) > 1:  # checks with Enchant library to give a bool value
                    english_words.append(word)
        return english_words
        #self.remove_stopwords(english_words)

    def remove_stopwords(self, words):  # here "words" is tokenized array
        stop_words = set(stopwords.words("english"))
        # with open('punc_and_ocr.txt', 'r') as stops:
        #     s = stops.read()
        #     stop_words_from_file = s.split()
        #
        # stop_words = set(stop_words_from_file)
        # print(stop_words)
        filered_words = []
        for word in words:
            if word not in stop_words:  # "len(words)<2" removes fullstops & other characters
                filered_words.append(word)
        return filered_words

    def lemmatizing(self, words):
        lemmatizer = WordNetLemmatizer()
        # l_words = map(lemmatizer.lemmatize(words))
        l_words = map(lemmatizer.lemmatize, words)
        # print l_words

    def stemming(self, words):  # here "words" is tokenized array
        stemmer = PorterStemmer()
        for w in words:
            stemmer.stem(w)

    def pos_tagging(self, words):
        tagged_words = pos_tag(words)

    def penn_to_wordnet(treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def clean_dataset(self):
        input_file = open('neg_CSV_URL.csv', 'rt',encoding='utf-8')
        l_p = LanguageProcessing()
        tokeniz = l_p.tokeniz(input_file)
        cleaned_url = l_p.clean_URL(tokeniz)
        remove_words = l_p.remove_non_englishwords(cleaned_url)
        stopwords_removed = l_p.remove_stopwords(remove_words)
        input_file.close()
        print (stopwords_removed)
        output_file=open('NewchromehistoryNEG_log.txt', 'wt',encoding='utf-8')
        output_file.write(' '.join(str(s) for s in stopwords_removed))
        output_file.close()

def main():
    l_p = LanguageProcessing()
    l_p.clean_dataset()


main()