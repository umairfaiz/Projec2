import sys,os

from PyQt4 import QtCore, QtGui
from New_Home_ASB_Tool import Ui_MainWindow
from generateSummary import summary

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

        #call addPiechart() and addWordCloud(self) afer the new deetails are generated

    def addPiechart(self):

        self.ui.lblPie.setPixmap(QtGui.QPixmap(os.getcwd() + "/polarity5.png"))
        print ("Added")

    def addWordCloud(self):

        self.ui.lblWordscloud.setPixmap(QtGui.QPixmap(os.getcwd() + "/WC.png"))
        print ("Added")

    def addDetails(self):
        number=self.ui.txtnumber.text()
        summaryinstance=summary()
        summaryinstance.insertNumber(number)
        self.ui.txtnumber.clear()
        self.ui.lblSuccess.setText("Added successfully! Now you will get a summary of the prediction to your mobile!")
        print(number)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())