import os
import sqlite3
from datetime import datetime, timedelta
from twilio.rest import Client
import datetime as dt
from bigram_classification import BigramsClassifier
from browser import Browser


class summary(object):

    def __init__(self):
        self.checkBrowserActivity()

    def checkBrowserActivity(self):
        connection = sqlite3.connect(os.getenv("APPDATA") + '\..\Local\Google\Chrome\\User Data\Default\history')
        connection.text_factory = str
        cur = connection.cursor()
        epoch = datetime(1601, 1, 1)

        for row in (cur.execute('select last_visit_time from urls')):
            row = list(row)
            url_time = epoch + timedelta(microseconds=row[0])
            new_url_time = url_time.strftime('%Y-%m-%d')
            now_time = url_time.now().strftime('%Y-%m-%d')

        if (new_url_time == now_time): # or now_time=='2017-10-15'
            # when true starts the to analyze the log automatically to send report to user
            Browser()
            BigramsClassifier()
            self.sendSummary()
        else:
            self.sendFailedSMS()

    def getUserNumber(self):
        with open('C:\\Users\Admin\PycharmProjects\FYP_CB006302\\userDetails.txt','r') as file:
            for line in file:
                parts = line.split(',')
                number = parts[0]
                #print(number)
        return number

    # def insertNumber(self,number):
    #     with open("userDetails.txt", "w") as text_file:
    #         mob_number="+94"+number
    #         text_file.write(mob_number)

    def sendFailedSMS(self):

        todayDate = str(dt.date.today().strftime("%d-%m-%y"))
        # percentage = str(negPercentage)
        mobile_number = self.getUserNumber()
        while True:
            # Your Account SID from twilio.com/console
            account_sid = "A"
            # Your Auth Token from twilio.com/console
            auth_token = "f"

            try:
                client = Client(account_sid, auth_token)

                message = client.messages.create(
                    to=mobile_number,
                    from_="+12547915242",
                    body="Hello! There was nothing browsed as of " + todayDate + ". We will let you know when the log is updated!.")

                print(message.sid)
                # print("MESSAGE FAILED")

            except:
                pass
            else:
                break


    def sendSMS(self, negPercentage):

        todayDate = str(dt.date.today().strftime("%d-%m-%y"))
        #percentage = str(negPercentage)
        mobile_number = self.getUserNumber()
        while True:
            # Your Account SID
            account_sid = "A"
            # Your Auth Token
            auth_token = "f"

            try:
                client = Client(account_sid, auth_token)

                message = client.messages.create(
                    to=mobile_number,
                    from_="+12547915242",
                    body="Hello! This is the summary as of " + todayDate + ". Where we predicted today's negative content to be " + negPercentage +"%.")

                print(message.sid)
                #print("MESSAGE SENT")
            except:
                pass
            else:
                break

    def sendSummary(self):

        with open('historylogs.txt', 'r') as f1:
            last_line = f1.readlines()[-1]
            parts = last_line.split(',')
            number = parts[1]
        self.sendSMS(number)

if __name__ == "__main__":
    summary()