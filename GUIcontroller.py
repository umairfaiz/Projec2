import sys,os

from PyQt4 import QtCore, QtGui
from New_Home_ASB_Tool import Ui_MainWindow
from generateSummary import summary
from bigram_classification import BigramsClassifier
from analysis import analyze

class MyForm(QtGui.QMainWindow):

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.lblHistory.setPixmap(QtGui.QPixmap(os.getcwd() + "/historygraph.png"))
        self.ui.pushButton.clicked.connect(self.generateDetails)
        # self.ui.pushButton.clicked.connect(self.addPiechart)
        # self.ui.pushButton.clicked.connect(self.addWordCloud)
        self.ui.btnsubmitsettings.clicked.connect(self.addDetails)
        #self.ui.connect(self.addHistoryGraph)



    def generateDetails(self):

        #call addPiechart() and addWordCloud(self) after the new details are generated
        #instancAnalysis= analyze()
        BigramsClassifier()
        #instanceBigramClassification.biClassify()

        self.addPiechart()
        self.addWordCloud()
        self.summaryText()

    def addPiechart(self):

        self.ui.lblPieName.setText("Predicted polarities")
        self.ui.lblPie.setPixmap(QtGui.QPixmap(os.getcwd() + "/polarity.png"))
        self.ui.lblgreen.setStyleSheet(' background-color : green')
        self.ui.lblpostiv.setText("Positive")
        self.ui.lblred.setStyleSheet(' background-color : red')
        self.ui.lblNegtv.setText("Negative")

        print ("Added")

    def addWordCloud(self):
        self.ui.lblWCName.setText("Most used words")
        self.ui.lblWordscloud.setPixmap(QtGui.QPixmap(os.getcwd() + "/WordCloud.png"))
        print ("Added")

    def addDetails(self):
        number=self.ui.txtnumber.text()
        summaryinstance=summary()
        summaryinstance.insertNumber(number)
        self.ui.txtnumber.clear()
        self.ui.lblSuccess.setText("Added successfully! Now you will get a summary of the prediction to your mobile on daily basis!")
        print(number)

    def summaryText(self):
        # summary = "We are "+str(confidence) +"% confident that "+ polarity+" negative words have been predicted. Just keep and eye!"
        # self.ui.label_5.setText(summary)
        self.ui.label_5.setPixmap(QtGui.QPixmap(os.getcwd() + "/summary_text.png"))
        print("Added")


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())