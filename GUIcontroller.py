import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import QThread
from PyQt4.QtGui import QMessageBox
from New_Home_ASB_Tool import Ui_MainWindow
from bigram_classification import BigramsClassifier
from browser import *

class MyForm(QtGui.QMainWindow):

    def __init__(self, parent=None):

        QtGui.QWidget.__init__(self, parent)
        QtGui.QWidget.setWindowFlags(self, QtCore.Qt.CustomizeWindowHint)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        Close_Action = QtGui.QAction("&Close tool", self)
        Close_Action.setShortcut("Ctrl+Q")
        Close_Action.triggered.connect(self.close_application)
        self.statusBar()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('&CLOSE')
        fileMenu.addAction(Close_Action)

        self.ui.lblHistory.setPixmap(QtGui.QPixmap(os.getcwd() + "/historygraph.png"))
        self.workerThread = WorkingThread()
        self.ui.pushButton.clicked.connect(self.generateDetails)
        self.workerThread.finished.connect(self.addgenerateDetails)

        # self.ui.pushButton.clicked.connect(self.addPiechart)
        # self.ui.pushButton.clicked.connect(self.addWordCloud)
        self.ui.btnsubmitsettings.clicked.connect(self.addDetails)
        #self.ui.connect(self.addHistoryGraph)
    def close_application(self):
        self.hide()

    def generateDetails(self):
        self.selectBrowser()
        if not self.workerThread.isRunning():
            self.workerThread.start()

    def addgenerateDetails(self):

        #call addPiechart() and addWordCloud(self) after the new details are generated
        #instancAnalysis= analyze()

        # self.selectBrowser()
        # BigramsClassifier()

        #instanceBigramClassification.biClassify()
        self.ui.pushButton.hide()
        self.ui.lblHistory.setPixmap(QtGui.QPixmap(os.getcwd() + "/historygraph.png"))
        self.addPiechart()
        self.addWordCloud()
        self.summaryText()
        # self.categoryText()

    def addPiechart(self):

        self.ui.lblPieName.setText("Predicted polarities of links")
        self.ui.lblPie.setPixmap(QtGui.QPixmap(os.getcwd() + "/polarity.png"))
        self.ui.lblgreen.setStyleSheet(' background-color : green')
        self.ui.lblpostiv.setText("Positive")
        self.ui.lblred.setStyleSheet(' background-color : red')
        self.ui.lblNegtv.setText("Negative")

        #print ("Added")

    def addWordCloud(self):
        self.ui.lblWCName.setText("Most used words")
        self.ui.lblWordscloud.setPixmap(QtGui.QPixmap(os.getcwd() + "/WordCloud.png"))
        #print ("Added")

    def insertNumber(self, number):

        with open("userDetails.txt", "w") as text_file:
            mob_number = "+94" + str(number)
            text_file.write(mob_number)


    def addDetails(self):
        number=self.ui.txtnumber.text()
        # summaryinstance=summary()
        # summaryinstance.insertNumber(number)
        # self.ui.txtnumber.clear()
        # self.ui.lblSuccess.setText("Added successfully! Now you will get a summary of the prediction to your mobile on daily basis!")

        while True:
            try:
                number1 = int(number)
                if len(number) == 9:
                    self.insertNumber(number1)
                    self.ui.txtnumber.clear()
                    self.ui.lblSuccess.setText(
                        "Added successfully! Now you will get a summary of the prediction to your mobile on daily basis!")
                    break
                else:
                    self.ui.lblSuccess.setText(
                        "Incorrect number format!")
                    # ctypes.windll.user32.MessageBoxW(None, "Incorrect number format!!", "Error", 0)
                    self.ui.txtnumber.clear()
                    break
            except ValueError:
                ctypes.windll.user32.MessageBoxW(None, "Only numbers allowed. Try again!", "Error", 0)
                self.ui.txtnumber.clear()
                break

    def summaryText(self):
        # summary = "We are "+str(confidence) +"% confident that "+ polarity+" negative words have been predicted. Just keep and eye!"
        # self.ui.label_5.setText(summary)
        self.ui.label_5.setPixmap(QtGui.QPixmap(os.getcwd() + "/summary_text.png"))
        #print("Added")

    # def categoryText(self):
    #
    #     self.ui.lblCategory.setPixmap(QtGui.QPixmap(os.getcwd() + "/category_text.png"))
    #     #print("Added")

    def browserError(self, error):
        print("Error from GUI",error)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setWindowTitle("Error")
        msg.show()

    def selectBrowser(self):
        selection = QtGui.QMessageBox.information(self, "Select Chrome Browser","YES- for chrome, No-For FireFox",QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        instanceBrowser=Browser()
        instanceFFBrowser=fireFoxBrowser()

        if selection == QtGui.QMessageBox.Yes:
            instanceBrowser.get_log()
            # sys.exit()
        else:
            instanceFFBrowser.get_fireFox_log()

class WorkingThread(QThread):
    def __init__(self, parent=None):
        super(self.__class__, self).__init__(parent)

    # q=Queue()
    def run(self):
        BigramsClassifier()
        #time.sleep(2)

if __name__ == "__main__":

    #try:
    app = QtGui.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    sys.exit(app.exec_())
    # except sqlite3.OperationalError as e:
    #     MyForm.browserError(e)
