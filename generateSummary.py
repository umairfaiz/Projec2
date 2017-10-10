import os
import csv
import sqlite3
from datetime import datetime, timedelta
from twilio.rest import Client
import datetime as dt


class summary(object):

    def __init__(self):
        pass

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

        if (new_url_time == now_time):
            # when true starts the to analyze the log automatically to send report to user
            print("Found!")
        else:
            print("Failed")

    def getUserNumber(self):
        with open('userDetails.txt') as file:
            for line in file:
                parts = line.split(',')
                number = parts[0]
                # print(number)
        return number

    def insertNumber(self,number):
        with open("userDetails.txt", "w") as text_file:
            mob_number="+94"+number
            text_file.write(mob_number)


    def sendSMS(self, negPercentage):
        todayDate = str(dt.date.today().strftime("%d-%m-%y"))
        percentage = str(negPercentage)
        mobile_number = self.getUserNumber()
        while True:
            # Your Account SID from twilio.com/console
            account_sid = "AC8287f4cca84f0e7c47d62ea6eb81a16c"
            # Your Auth Token from twilio.com/console
            auth_token = "f43c69f462496b12d4d05cea68b26446"

            try:
                client = Client(account_sid, auth_token)

                message = client.messages.create(
                    to=mobile_number,
                    from_="+12547915242",
                    body="Hello! This is the summary as of " + todayDate + ". Where we predicted today's negative content to be " + percentage + "%.")

                print(message.sid)
            except:
                pass
            else:
                break

    def sendSummary(self, percentage):

        newPercentage = percentage * 100
        self.sendSMS(newPercentage)
