from bigram_classification import bi_features
import matplotlib.pyplot as plt
from os import path
from wordcloud import WordCloud
import numpy as np

import datetime as dt

class analyze(object):
    def __init__(self):
        pass

    def polarity_analysis(self, classifier, file):
        results = {"Positive": 0, "Negative": 0}
        number_sentences = 0
        for sentence in file:
            number_sentences += 1
            line = sentence.replace("\n", "").replace(".", "")
            polarity = classifier.classify(bi_features(line))
            if str(polarity) == 'pos':
                results["Positive"] += 1
            elif str(polarity) == 'neg':
                results["Negative"] += 1
        polarity_pos_rate = (results["Positive"] / number_sentences)
        polarity_neg_rate = (results["Negative"] / number_sentences)
        print("Positive percentage: ", polarity_pos_rate * 100)
        print("Negative percentage: ", polarity_neg_rate * 100)
        analyze.plot_piechart(polarity_pos_rate, polarity_neg_rate)

    def plot_piechart(self, p, n): ###called in polarity_analysis()

        labels = 'Positive', 'Negative'
        sizes = [p, n]
        colors = ['green', 'red']
        explode = (0.1, 0)  # explode 1st slice

        # Plotting pie chart
        plt.figure(figsize=(4, 2))  # getting a figure with dimension in inches
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)

        plt.axis('tight')
        plt.savefig('polarity.png', bbox_inches='tight')
        plt.show()

    def creatWordCloud(self,document):

        d = path.dirname(__file__)
        doc=document+'.txt'
        # Read the whole text.
        text = open(path.join(d, doc)).read()

        # Generate a word cloud image
        wordcloud = WordCloud(background_color="white", ).generate(text)

        plt.figure(figsize=(4, 2))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('WordCloud.png', bbox_inches='tight')
        plt.show()

    def addDatatoFile(percentage):
        today_date = dt.date.today().strftime("%d")
        # print(today_date)
        with open("historylogs.txt", mode='a') as file:
            file.write('%s,%s\n' %(today_date, percentage))
            file.close()


    def createHistoryGraph(self):
        #plotting a graph from a file with only negative percentages everyday
        x,y=np.loadtxt('historylogs.txt', delimiter=',', unpack=True)
        plt.figure(figsize=(4, 2))
        plt.plot(x,y, label='Negative percentage')
        plt.xlabel('Date')
        plt.ylabel('Percentage (%)')
        plt.legend()
        plt.savefig('historygraph.png', bbox_inches='tight')
        plt.show()




def main():
    analysis = analyze()
    analysis.polarity_analysis()

main()