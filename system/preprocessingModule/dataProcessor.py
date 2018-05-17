
# coding: utf-8
# encoding=utf8
import sys

reload(sys)
sys.setdefaultencoding('utf8')



import pandas as pd
from string import punctuation
import time

FILENAME = "Tweets2.csv"

df_btc_1 = pd.read_csv('bitcoin data/twitter_btc_1.csv',delimiter='\t')
df_btc_2 = pd.read_csv('bitcoin data/twitter_btc_2.csv',delimiter='\t')
df_btc_3 = pd.read_csv('bitcoin data/twitter_btc_3.csv',delimiter='\t')
df_btc_4 = pd.read_csv('bitcoin data/twitter_btc_4.csv',delimiter='\t')
df_btc_5 = pd.read_csv('bitcoin data/twitter_btc_5.csv',delimiter='\t')
df_btc_6 = pd.read_csv('bitcoin data/twitter_btc_6.csv',delimiter='\t')
df_btc_7 = pd.read_csv('bitcoin data/twitter_btc_7.csv',delimiter='\t')
df_btc_8 = pd.read_csv('bitcoin data/twitter_btc_8.csv',delimiter='\t')
df_btc_9 = pd.read_csv('bitcoin data/twitter_btc_9.csv',delimiter='\t',error_bad_lines=False, encoding = "ISO-8859-1")
df_btc_10 = pd.read_csv('bitcoin data/twitter_btc_10.csv',delimiter='\t')

def assignColums(columns):
    df_btc_1.columns = columns
    df_btc_2.columns = columns
    df_btc_3.columns = columns
    df_btc_4.columns = columns
    df_btc_5.columns = columns
    df_btc_6.columns = columns
    df_btc_7.columns = columns
    df_btc_8.columns = columns
    df_btc_9.columns = columns
    df_btc_10.columns = columns

def appendCSVs(csvs):
    df_btc = df_btc_1.append(csvs, ignore_index=True)
    return df_btc


def clean_tweets(df):
    df_clean = df[df.tweet.str.contains('giveaway') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('give away') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('give-away') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('retweet this and give one to each') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('giving away') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('giving away') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('The most complete and provably fair #Bitcoin #Lottery') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('As soon as you sign up, you can earn more money') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('give away') == False]
    df_clean = df_clean[df_clean.tweet.str.contains('RT ') == False]
    return df_clean

def convertToLowercase(df_btc):
    return df_btc['tweet'].str.lower()

def removePunctuations(df_btc):
    return df_btc.tweet2.apply(lambda x: ''
                                     .join(i if i not in punctuation else '' for i in x))
def removeLinks(df_btc):
    return df_btc.tweet2.apply(lambda x: ' '.join(['' if 'http' in i or len(i) > 25 else i
                                                        for i in x.split(' ')]))

def removeExcessWhiteSpace(df_btc):
    return df_btc.tweet2.apply(lambda x: ' '.join(list(filter(None, x.split(' ')))))

def assignTweetLength(df_btc):
    df_btc['tweet_length'] = df_btc['tweet2'].str.split(" ")
    df_btc['tweet_length'] = df_btc['tweet_length'].apply(lambda x: len(x))
    return df_btc.loc[df_btc['tweet_length'] > 2]

def assignTimeDifference(df_btc):
    time_dif = df_btc['dtime'][1:].reset_index(drop=True) - df_btc['dtime'][:-1].reset_index(drop=True)
    df_btc = df_btc.reset_index(drop=True)
    time_dif_1 = pd.to_timedelta(pd.Series(['0'])).append(time_dif).reset_index(drop=True)
    df_btc['time dif'] = time_dif_1

    return df_btc

def removeNewLineCharacters(df_btc):
    df_btc['tweet3'] = df_btc.tweet2
    df_btc['tweet3'] = df_btc.tweet3.apply(lambda x: ' '.join(x.encode('utf-8').split('\n')))
    df_btc['tweet3'] = df_btc.tweet3.apply(lambda x: ' '.join(x.split('\t')))
    df_btc['tweet3'] = df_btc.tweet3.apply(lambda x: ' '.join(x.split('\r')))

    return df_btc

#Start tracking time of the whole procedure:
global_start_time = time.time()

columns = ['tweet_id','tweet','dtime','user_id']
assignColums(columns)

df_btc = appendCSVs([
    df_btc_2,
    df_btc_3,
    df_btc_4,
    df_btc_5,
    df_btc_6,
    df_btc_7,
    df_btc_8,
    df_btc_9,
    df_btc_10])

print "Shape of new data frame of tweets: " , df_btc.shape
print ""

#Drop duplicates
print "Dropping duplicates..."
start_time = time.time()

df_btc = df_btc.drop_duplicates('tweet')

elapsed_time = time.time() - start_time
print "Duplicates dropped in " , time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print "Shape after droping duplicates: " , df_btc.shape
print ""
#Clean tweets
print "Cleaning tweets..."
start_time = time.time()

df_btc = clean_tweets(df_btc)

elapsed_time = time.time() - start_time

print "Tweets cleaned in ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""

#Convert to lowercase
print "Tweet before lowercase: '", df_btc['tweet'].iloc[0], "'"
print "Converting to lowercase..."

start_time = time.time()

df_btc['tweet2'] = convertToLowercase(df_btc)

elapsed_time = time.time() - start_time
print "Tweets converted to lowercase in", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print "Tweet after lowercase: '", df_btc['tweet2'].iloc[0], "'"
print ""
#Remove punctuations
print "Tweet before punctuations: '", df_btc['tweet'].iloc[0], "'"
print "Removing punctuations..."

start_time = time.time()

df_btc['tweet2'] = removePunctuations(df_btc)

elapsed_time = time.time() - start_time
print "Punctuation done in ",  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print "Tweet after punctuations: '", df_btc['tweet2'].iloc[0], "'"
print ""
#Remove links
print "Removing links..."
start_time = time.time()

df_btc['tweet2'] = removeLinks(df_btc)

elapsed_time = time.time() - start_time
print "Remove links done in", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#Remove excess whitespace
print "Removing whitespace..."
start_time = time.time()

df_btc['tweet2'] = removeExcessWhiteSpace(df_btc)

elapsed_time = time.time() - start_time
print "Remove excess whitespace done in ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#Assign tweet length to each tweet, remove short ones
print "Assigning tweet length..."
start_time = time.time()

df_btc = assignTweetLength(df_btc)

elapsed_time = time.time() - start_time
print "Assign tweet length done in ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#Converting date
print "Converting date..."
start_time = time.time()

df_btc['dtime'] = pd.to_datetime(df_btc.dtime)

elapsed_time = time.time() - start_time
print "Converting date done in", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""

#Sorting by date
print "Sorting by date..."
start_time = time.time()

df_btc = df_btc.sort_values('dtime')

elapsed_time = time.time() - start_time
print "Sorting bby date done in", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#Assign time difference
print "Assigning time difference..."
start_time = time.time()

df_btc = assignTimeDifference(df_btc)

elapsed_time = time.time() - start_time
print "Assigning time difference done in ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#sort by time difference
print "Sorting by time difference..."
start_time = time.time()

df_btc.sort_values('time dif', ascending=False).head(100)

elapsed_time = time.time() - start_time
print "Sorting by time difference done in ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#Remove new line characters
print "Removing new line characters..."
start_time = time.time()

df_btc = removeNewLineCharacters(df_btc)

elapsed_time = time.time() - start_time
print "Removing new line characters done in", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""
#write to file
print "Writing to csv and sorting..."
start_time = time.time()

df_btc_write = df_btc[['tweet_id', 'tweet3','dtime']]
df_btc_write.to_csv(FILENAME, index=False, header=False,sep='\t')
print ""
#sort values
df_btc_write.sort_values('tweet3')

elapsed_time = time.time() - start_time
print "Writing and sorting done in ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
print ""

#Global running time
global_elapsed_time = time.time() - global_start_time
print "Everythin done in ", time.strftime("%H:%M:%S", time.gmtime(global_elapsed_time))

