# from bigram_classification import bi_features
from select import select

import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud
import numpy as np
# from bigram_classification import BigramsClassifier
from nltk.stem import WordNetLemmatizer
import datetime as dt
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os


class analyze(object):
    features_method = None
    finalpol = None
    neg_perc = None
    confi = None

    def __init__(self, features_method, finalpol, neg_perc, confi):
        self.features_method = features_method
        self.finalpol = finalpol
        self.neg_perc = neg_perc
        self.confi = confi

        # self.polarity_analysis(classifier, file)
        # self.plot_piechart()
        # self.createWordCloudfile()
        # self.creatWordCloud()
        # self.addDatatoFile()
        self.createHistoryGraph(neg_perc)
        self.categorize_user()
        # self.summaryPolarity()


    # def polarity_analysis(self, classifier, file):
    #     results = {"Positive": 0, "Negative": 0}
    #     number_sentences = 0
    #     for sentence in file:
    #         number_sentences += 1
    #         line = sentence.replace("\n", "").replace(".", "")
    #         polarity = classifier.classify(self.features_method(line))
    #         if str(polarity) == 'pos':
    #             results["Positive"] += 1
    #         elif str(polarity) == 'neg':
    #             results["Negative"] += 1
    #     polarity_pos_rate = (results["Positive"] / number_sentences)
    #     polarity_neg_rate = (results["Negative"] / number_sentences)
    #     print("Positive percentage: ", polarity_pos_rate * 100)
    #     print("Negative percentage: ", polarity_neg_rate * 100)
    #     analyze.plot_piechart(polarity_pos_rate, polarity_neg_rate)
    #     print("polarity analyysis")

    def plot_piechart(self, p, n):

        # labels = 'Positive', 'Negative'
        sizes = [p, n]
        colors = ['green', 'red']
        explode = (0.1, 0)  # explode 1st slice

        # Plotting pie chart
        plt.figure(figsize=(4, 2))  # getting a figure with dimension in inches
        plt.pie(sizes, explode=explode, colors=colors,autopct='%1.1f%%', shadow=True, startangle=90)

        plt.axis('tight')
        plt.savefig('polarity.png', bbox_inches='tight')
        print("Pie-chart added")
        print(self.finalpol, self.neg_perc, self.confi)
        # plt.show()

    def createWordCloudfile(self, neg_sentnces):

        with open('wordcloudfile.txt', 'w')as file:
            for sentence in neg_sentnces:
                file.write(sentence + '\n')

                # def creatWordCloud(self, document):
        d = path.dirname(__file__)
        # Read the whole text.
        text = open(path.join(d, 'wordcloudfile.txt')).read()
        if text=='':
            text='NONE'
            wordcloud = WordCloud(background_color="white").generate(text)
        else:
        # Generate a word cloud image
            wordcloud = WordCloud(background_color="white").generate(text)

        plt.figure(figsize=(4, 2))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('WordCloud.png', bbox_inches='tight')
        # plt.show()
        print("WordCloud added")

    # def categoryGeneration(self, sentence):
    #
    #     fonts_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
    #     font_name = 'consolab.ttf'
    #     font = ImageFont.truetype(os.path.join(fonts_dir, font_name), 15)
    #     d = path.dirname(__file__)
    #     # Opening the file gg.png
    #     imageFile = d + "/white_correct_res.png"
    #     im1 = Image.open(imageFile)
    #
    #     # Drawing the text on the picture
    #     draw = ImageDraw.Draw(im1)
    #     draw.text((20, 10), sentence, (0, 0, 0), font=font)
    #     draw = ImageDraw.Draw(im1)
    #
    #     # Save the image with a new name
    #     im1.save("category_text.png")
    #     print("category added")

    def categorize_user(self):

        category_dict = {"Death": 0, "Terrorism": 0, "Racism":0}

        Death_words=['suicide','murder','kill','death','die','stab', 'dead']
        terrorism_words=['bombing','bomb','gun','battle', 'war','terrorist']
        racism_words=['nigga','racist','hate','black']

        results = []
        with open('C:\\Users\Admin\PycharmProjects\FYP_CB006302\\predictedNegativefile.txt', 'r') as f:
            for line in f:
                line.lower()
                results.extend(line.strip().split())

        lemmatizer = WordNetLemmatizer()

        lemma = list(map(lemmatizer.lemmatize, results))

        for word in lemma:
            if word in Death_words:
                category_dict["Death"]+= 1
            elif word in terrorism_words:
                category_dict["Terrorism"] += 1
            elif word in racism_words:
                category_dict["Racism"] += 1
            else:
                pass
        categ = []
        for key, value in category_dict.items():

            if value > 0:
                categ.append(key)
            else:
                pass

        sent = ', '.join(categ)
        if sent=='':
            category_sentence="-"
        else:
            category_sentence="Our prediction is that user is more biased to the category(s) : "+sent+"."
            #print("category set to print")
        return category_sentence

        #self.categoryGeneration(category_sentence)

    def createHistoryGraph(self, percentage):
        new_percentage=round(percentage)
        today_date = dt.date.today().strftime("%d")
        # print(today_date)
        with open("historylogs.txt", mode='a') as file:
            file.write('%s,%s\n' % (today_date, new_percentage))
            file.close()
        # plotting a graph from a file with only negative percentages everyday
        x, y = np.loadtxt('historylogs.txt', delimiter=',', unpack=True)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Negativity')
        plt.xlabel('Date')
        plt.ylabel('Percentage (%)')
        plt.legend()
        plt.savefig('historygraph.png', bbox_inches='tight')
        # plt.show()
        print("History graph added")
        # analyze.summaryPolarity(self)

    # def addDatatoFile(self, percentage):

    def summaryGeneration(self, sentence):

        summary=sentence+self.categorize_user()
        #print("this->",self.categorize_user())
        fonts_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        font_name = 'consolab.ttf'
        font = ImageFont.truetype(os.path.join(fonts_dir, font_name), 15)
        d = path.dirname(__file__)
        # Opening the file gg.png
        imageFile = d+"/white_correct_res.png"
        im1 = Image.open(imageFile)

        # Drawing the text on the picture
        draw = ImageDraw.Draw(im1)
        draw.text((20, 10), summary, (0, 0, 0), font=font)
        draw = ImageDraw.Draw(im1)

        # Save the image with a new name
        im1.save("summary_text.png")

    def summaryPolarity(self, polarity, negative_rate, confidence_perc):

        confidence_perc=int(confidence_perc)
        # polarity = self.finalpol
        # negative_rate = self.neg_perc
        # confidence_perc = self.confi
        if negative_rate == 0.00 and polarity == "pos":
            sentence = "We are " + str(
                confidence_perc) + "% confident that no negative words have been predicted. However, Just keep an eye!\n"
            self.summaryGeneration(sentence)
            print("1")
        elif negative_rate <= 0.30 and polarity == "pos":
            sentence = "We are " + str(
                confidence_perc) + "% confident that mild negative words have been predicted. Just keep an eye!\n"
            self.summaryGeneration(sentence)
            print("1")
        elif negative_rate > 0.30 and polarity == "pos":
            sentence = "We are " + str(
                confidence_perc) + "% confident that strong negative words have been predicted. Just keep an eye!\n"
            self.summaryGeneration(sentence)
            print("2")
        elif negative_rate > 0.40 and polarity == "neg":
            sentence = "We are " + str(
                confidence_perc) + "% confident that very strong negative words have been predicted. Just keep an eye!\n"
            self.summaryGeneration(sentence)
            print("3")
        elif negative_rate > 0.00 and polarity == "neg":
            sentence = "We are " + str(
                confidence_perc) + "% confident that strong negative words have been predicted. Just keep an eye!\n"
            self.summaryGeneration(sentence)
            print("3")
        else:
            sentence = "Not enough data to predict user behavior. Try again after few days!\n"
            self.summaryGeneration(sentence)

    # def summaryPolarity(self):
    #     from GUIcontroller import MyForm
    #
    #     instanceMyForm=MyForm()
    #
    #     # polarity = self.finalpol
    #     negative_rate = self.neg_perc
    #     # confidence_perc = self.confi
    #
    #     if negative_rate <= 51 and self.finalpol == "pos":
    #         instanceMyForm.summaryText("mild", self.confi)
    #         print("1")
    #     elif negative_rate > 51 and self.finalpol == "pos":
    #         instanceMyForm.summaryText("strong", self.confi)
    #         print("2")
    #     elif negative_rate > 51 and self.finalpol == "neg":
    #         instanceMyForm.summaryText("very strong", self.confi)
    #         print("3")

# def main():
#     analysis = analyze()
#     analysis.polarity_analysis()
#
#
# main()
