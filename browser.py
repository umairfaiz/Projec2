import codecs
import csv
import os
import sqlite3


class Browser(object):
    def __init__(self):
        self.get_log()

    def get_log(self):
        connection = sqlite3.connect(os.getenv("APPDATA")+'\..\Local\Google\Chrome\\User Data\Default\history')
        connection.text_factory = str
        cur = connection.cursor()
        try:
            #with codecs.open('chromehistory_log.txt', 'wb',encoding='utf-8')as output_file:
            output_file = open('chromehistory_log.csv', 'wt', encoding='utf-8')  # wb - write binary wt-write text
            #codecs.encode(output_file,'utf-8')
            csv_writer = csv.writer(output_file)
            headers = ('URL', 'Title', 'Visit Count')
            csv_writer.writerow(headers)
            for row in (cur.execute('select url, title, visit_count from urls')):
                row = list(row)
                csv_writer.writerow(row)
        finally:
            output_file.close()

def main():
    browser = Browser()
    browser.get_log()

    # l_p = LanguageProcessing()
    # l_p.tokeniz()

main()
