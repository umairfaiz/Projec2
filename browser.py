import csv
import os
import sqlite3
import ctypes
from collections import deque


class Browser(object):
    def __init__(self):
        self.get_log()

    def get_log(self):
        connection = sqlite3.connect(os.getenv("APPDATA")+'\..\Local\Google\Chrome\\User Data\Default\history')
        connection.text_factory = str
        cur = connection.cursor()

        try:

            #with codecs.open('chromehistory_log.txt', 'wb',encoding='utf-8')as output_file:
            output_file = open('chromehistory_log.txt', 'w', encoding='utf-8',newline='')  # wb - write binary wt-write text
            csv_writer = csv.writer(output_file)
            for row in (cur.execute('select url, title, visit_count from urls')):
                csv_writer.writerow(list(row))
            output_file.close()

            with open('chromehistory_log.txt', encoding='utf-8') as fin, open('limitLogHistory_log.txt', 'w', encoding='utf-8') as fout:
                fin.seek(0)
                if not fin.read(1):
                    ctypes.windll.user32.MessageBoxW(None, "No browser content!", "Error", 0)
                    exit(0)
                else:
                    fout.writelines(deque(fin, 200))

        except sqlite3.OperationalError as e:
            print("Error from browser",e)
            ctypes.windll.user32.MessageBoxW(None, "Please close your browser!", "Error", 0)
            exit(0)

class fireFoxBrowser(object):

    def __init__(self):
        self.get_fireFox_log()

    def get_fireFox_log(self):
        try:
            data_path = os.getenv("APPDATA") + "\\Mozilla\\Firefox\\Profiles\\upat5xii.default"
            files = os.listdir(data_path)
            history_db = os.path.join(data_path, 'places.sqlite')

            c = sqlite3.connect(history_db)
            cursor = c.cursor()
            select_statement = "select moz_places.url, moz_places.title, moz_places.visit_count from moz_places;"
            cursor.execute(select_statement)

            output_file = open('firefoxhistory_log.txt', 'w', encoding='utf-8',
                               newline='')  # wb - write binary wt-write text
            csv_writer = csv.writer(output_file)
            for row in (cursor.execute(select_statement)):
                csv_writer.writerow(list(row))

            output_file.close()

            with open('firefoxhistory_log.txt', encoding='utf-8') as fin, open('limitLogHistory_log.txt', 'w', encoding='utf-8') as fout:
                fin.seek(0)
                if not fin.read(1):
                    ctypes.windll.user32.MessageBoxW(None, "No browser content!", "Error", 0)
                    exit(0)
                else:
                    fout.writelines(deque(fin, 200))


        except sqlite3.OperationalError as e:
            print("Error from browser",e)
            ctypes.windll.user32.MessageBoxW(None, "Please close your FF browser!", "Error", 0)
            exit(0)

# def main():
#
#     browser = Browser()
#     browser.get_log()
#
#     # ff=fireFoxBrowser()
#     # ff.get_fireFox_log()
#
#     # l_p = LanguageProcessing()
#     # l_p.tokeniz()
#
# main()
