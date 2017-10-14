import csv
import os
import sqlite3
from collections import deque

class Browser(object):
    def __init__(self):
        self.get_log()

    def get_log(self):
        connection = sqlite3.connect(os.getenv("APPDATA")+'\..\Local\Google\Chrome\\User Data\Default\history')
        connection.text_factory = str
        cur = connection.cursor()

        #with codecs.open('chromehistory_log.txt', 'wb',encoding='utf-8')as output_file:
        output_file = open('chromehistory_log.txt', 'w', encoding='utf-8',newline='')  # wb - write binary wt-write text
        csv_writer = csv.writer(output_file)
        for row in (cur.execute('select url, title, visit_count from urls')):
            csv_writer.writerow(list(row))
        output_file.close()

        with open('chromehistory_log.txt', encoding='utf-8') as fin, open('limitchromehistory_log.txt', 'w', encoding='utf-8') as fout:
            fout.writelines(deque(fin, 100))


# def main():
#     browser = Browser()
#     browser.get_log()
#
#     # l_p = LanguageProcessing()
#     # l_p.tokeniz()
#
# main()
