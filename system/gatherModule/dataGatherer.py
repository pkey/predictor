# coding: utf-8
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')


import csv
from time import sleep
import datetime
import twitterApi as twitterApi

def extract_tweets(query, filename):
    while True:
        print('starting {0}'.format(datetime.datetime.now()))
        try:
            msg = twitterApi.search(query)
            for i in msg:
                keys = i.keys()
                if 'id' in keys and 'text' in keys and 'created_at' in keys and 'user' in keys:
                    line = []
                    line.append(i['id'])
                    line.append(i['text'])
                    line.append(i['created_at'])
                    line.append(i['user']['id'])
                    with open(filename, 'a') as fp:
                        writter = csv.writer(fp, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
                        writter.writerow(line)
        except Exception as e:
            print(e)
        print('sleeping {0}'.format(datetime.datetime.now()))
        sleep(30)


extract_tweets('bitcoin', 'bitcoin data/twitter_btc.csv')


