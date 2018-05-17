
# coding: utf-8


import numpy as np
import pandas as pd
import tensorflow as tf
from string import punctuation


def getTestDatasetFromCsv():
    tweets = pd.read_csv("data/Test.csv", delimiter=',', encoding='utf8')
    return tweets

def getTweetsFromCsv(filename="data/Tweets.csv"):
    tweets = pd.read_csv(filename, delimiter='\t', names=['id', 'text', 'time'], encoding='utf8')
    tweets['time'] = pd.to_datetime(tweets['time'])
    tweets = tweets.sort_values('time')
    return tweets

def getPricesFromCsv(pricefile = 'data/Prices.csv'):  
    return pd.read_csv(pricefile)

def matchPriceDelay(rawPrices, delay):
    #Get target price, applying delay
    rawPrices['target'] = rawPrices.loc[delay:, ['open']].reset_index(drop=True)
    
    #Get increased by substracting difference and applying boolean expression
    rawPrices['increased'] = rawPrices.loc[delay:, ['open']].reset_index(drop=True) - rawPrices.loc[:,['open']].reset_index(drop=True)
    rawPrices['increased'] = rawPrices['increased'].apply(lambda x: 1 if x > 0 else 0)
    return rawPrices

def getWordSet(tweets):
    #Get distinct set of words from messages
    return set([i for i in ' '.join(tweets).split(' ')])

def mapWordsToIntegers(words):
    #Create word to integer map
    return {word:index+1 for index,word in enumerate(words)}

def getCleanedTweets(tweets):
    #Removed punctuations and excess spaces
    cleanedTweets = []
    for tweet in tweets:
        cleanedTweets.append(''.join(
            [c for c in ' '.join(filter(None, tweet.split(' '))) if c not in punctuation]
        ))
    return cleanedTweets

def preprocessTweets(tweets, sequenceLength, tweets_ints):
    proccessedTweets = np.zeros((len(tweets), sequenceLength),dtype=int)
    for index, tweet in enumerate(proccessedTweets):
        tweet_int = tweets_ints[index]
        tweet[sequenceLength - len(tweet_int):] = tweet_int
    return proccessedTweets


def getPriceDictionary(prices):
    #Create dictionary of price date and whether it has increased or not
    return {row['start']:row['increased'] for index, row in prices.iterrows()}

def getTrainingData(tweets, price_map, delay=1):
    time_keys = sorted(list(price_map.keys()))[:-delay]
    tweets_train = tweets.loc[tweets['time'].astype(str).isin(time_keys)]
    labels_train = [price_map[str(row['time'] + pd.Timedelta(minutes=delay))] for index,row in tweets_train.iterrows()]
    return tweets_train, labels_train


def getLongestTweetLength(tweets):
    longest = 0
    for tweet in tweets:
        length = len(tweet.split(' '))
        if length > longest:
            longest = length
    return longest

def mapTweetsToWordIntegerMaps(tweets, wordIntegers):
    #map tweets messages to word integeres
    proccessedTweets = []
    for tweet in tweets:
        proccessedTweets.append([wordIntegers[word] for word in tweet.split(' ')
                                          if word != '' and word!= ' '])
    return proccessedTweets 
    
def splitData(data, labels, ratio):    
    #Split data
    pivotIndex = int(len(data)*ratio)
    train_x, val_x = data[:pivotIndex], data[pivotIndex:]
    train_y, val_y = labels[:pivotIndex], labels[pivotIndex:]

    testPivotIndex = int(len(val_x)*0.5)
    val_x, test_x = val_x[:testPivotIndex], val_x[testPivotIndex:]
    val_y, test_y = val_y[:testPivotIndex], val_y[testPivotIndex:]
    
    return train_x, train_y, val_x, val_y, test_x, test_y

def getAccuracy(labels, predictions):
    with tf.name_scope('accuracy'):        
        correct_pred = tf.equal(labels,tf.cast(tf.round(tf.sigmoid(predictions)), tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('accuracy',accuracy)
    return accuracy

def getCost(labels, predictions):
    with tf.name_scope('cost'):
        cost = tf.losses.mean_squared_error(labels, predictions)
        tf.summary.scalar('cost', cost)
    return cost

def getPredictions(embeding, lstm_size, lstm_layers,keep_prob, num_outputs=1):
    outputs = buildRnn(embeding, lstm_size, lstm_layers, keep_prob)

    with tf.name_scope('predictions'):
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1],
        num_outputs=num_outputs,activation_fn=None)
        tf.summary.histogram('predictions', predictions)
    return predictions

def getBatches(x, y, batch_size=100):    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        
def getLabels(output_dtype, batch_size):
    return tf.placeholder(output_dtype, shape=[batch_size, 1], name='labels')

def getInputs(output_dtype, batch_size, seq_len):
    return tf.placeholder(dtype=output_dtype, shape=[batch_size, seq_len ], name="inputs")

def getSequenceLength():
    return tf.placeholder(dtype=tf.float32,name='seq_len')

def getLearningRate():
    with tf.name_scope('learning_rate'):
        learning_rate = tf.placeholder(dtype=tf.float32, shape=(None), name="learning_rate")
        tf.summary.scalar('learning_rate', learning_rate)
    return learning_rate

def getDropoutKeepProbability():
    return tf.placeholder(tf.float32, name='keep_prob')

def getEmbedding(inputs, word_number, embedding_size):
    with tf.name_scope('embeding'):
        embeddings = tf.Variable(tf.random_uniform([word_number, embedding_size], -1, 1, seed=123))
        embedded_words = tf.nn.embedding_lookup(embeddings, inputs)
    return embedded_words

def getRnnCell(lstm_size, keep_prob):
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop

def buildRnn(inputs, lstm_size, lstm_layers, keep_prob):
    cell = tf.nn.rnn_cell.MultiRNNCell(
        [getRnnCell(lstm_size, keep_prob) for _ in range(lstm_layers)])
    rnn, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs, dtype=tf.float32)
    return rnn

def getOptimiser(lr, cost):            
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(lr).minimize(cost)
    return optimizer

def formatSummaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)        


# ### Load data

# #### Method for loading test dataset

# In[5]:


def loadTestDataset():
    tweets = getTestDatasetFromCsv()
    prices = tweets.airline_sentiment.apply(lambda x: 0 if x == 'negative' else 1)
    
    prices = np.array(prices).reshape(len(prices),1)
    print("Tweets dataset size: {0}".format(len(tweets)))
    
    word_list = getWordSet(tweets['text'])
    word_2_int = mapWordsToIntegers(word_list)
    tweets_ints = mapTweetsToWordIntegerMaps(tweets['text'], word_2_int)
    
    longest = getLongestTweetLength(tweets['text'])
    print("Total unique word count: {0}".format(len(word_list)))
    
    tweet_final = preprocessTweets(sequenceLength=longest, tweets=tweets_ints, tweets_ints=tweets_ints)
    
    n_words = len(word_2_int) + 1 
    
    return tweets, tweet_final, word_list, word_2_int, tweets_ints, n_words,longest,prices


# #### Method for loading actual dataset

# In[6]:


def loadActualDataset(latency, seq_len):
    tweets = getTweetsFromCsv()
    prices_raw = getPricesFromCsv() 
    
    prices_raw = matchPriceDelay(prices_raw, latency)  
    
    tweets['time'] = tweets['time'].apply(lambda x: x.replace(second=0))
    price_map = getPriceDictionary(prices_raw)
    
    tweets1, labels1 = getTrainingData(tweets, price_map, delay=1)
    
    prices = np.array(labels1).reshape(len(labels1),1)
    print("Tweets dataset size: {0}".format(len(tweets1)))
    
    tweets1['text2'] = tweets1.text.apply(lambda x: " ".join(x.split(" ")[-seq_len:]))
    
    cleaned_tweets = getCleanedTweets(tweets1['text2'])
    word_list = getWordSet(cleaned_tweets)
    word_2_int = mapWordsToIntegers(word_list)
    tweets_ints = mapTweetsToWordIntegerMaps(cleaned_tweets, word_2_int)    
    
    longest = getLongestTweetLength(cleaned_tweets)
    print("Total unique word count: {0}".format(len(word_list)))
    
    tweet_final = preprocessTweets(sequenceLength=longest, tweets=tweets_ints, tweets_ints=tweets_ints)
    
    n_words = len(word_2_int) + 1 
    
    return tweets1, tweet_final, word_list, word_2_int, tweets_ints, n_words,longest,prices    


# #### Method for splitting dataset into training, validation and test

# In[7]:


def getSplitDatasets(tweet_final, prices, is_logits = False):     
    train_x, train_y, val_x, val_y, test_x, test_y = splitData(tweet_final, prices, 0.8)    
    if is_logits:
        train_y = list(map(lambda x: [0,1] if x == 0 else [1,0], train_y))
        val_y = list(map(lambda x: [0,1] if x == 0 else [1,0], val_y))
        test_y = list(map(lambda x: [0,1] if x == 0 else [1,0], test_y))
    
    return train_x, train_y, val_x, val_y, test_x, test_y


# #### Method for plotting graph

# In[8]:


def plotGraph(tweets1):
    abc = tweets1.text.apply(lambda x: len(x.split(" ")))
    plot = abc.sort_values().reset_index(drop=True).plot(kind='line',)
    import matplotlib.pyplot as plt
    plt.ylabel("item length",size=14)
    plt.xlabel("tweet number", size=14)
    plt.show(plot)


# ### Parameters
# 

# #### Test run?

# In[9]:


testRun = False


# #### Experiment and model number

# In[10]:


#Experinment number
experimentNumber = 2

#Model number
modelNumber = 2


# ### Hyper parameters

# In[11]:


if testRun:
    RnnSize = 128
    RnnLayers = 1
    batchSize = 64
    sequenceLength = 36
    embeddingSize = 32
    learningRate = 0.001
    keepProbability = 0.7
    epochNumber = 20
else:
    RnnSize = 512
    RnnLayers = 3
    batchSize = 250
    sequenceLength = 33
    embeddingSize = 32
    learningRate = 0.001
    keepProbability = 0.7
    epochNumber = 1


# In[12]:


if testRun:
    tweets, tweet_final, word_list, word_2_int, tweets_ints, n_words, longest, prices = loadTestDataset()
else: 
    tweets, tweet_final, word_list, word_2_int, tweets_ints, n_words, longest,prices = loadActualDataset(
    1, sequenceLength)
    #plotGraph(tweets)

train_x, train_y, val_x, val_y, test_x, test_y = getSplitDatasets(tweet_final, prices)


# #### Shape of the dataset

# In[13]:


train_x.shape


# ### Build the network

# In[14]:


tf.reset_default_graph()

train_graph = tf.Graph()
with train_graph.as_default():
    inputs = getInputs(tf.int32, batchSize, sequenceLength)
    labels = getLabels(tf.int32,batchSize)
    lr = getLearningRate()
    drop_keep_porob = getDropoutKeepProbability()
    embed = getEmbedding(inputs=inputs, word_number=n_words, embedding_size=embeddingSize)
    
    predictions = getPredictions(embed, RnnSize, RnnLayers, drop_keep_porob)
    cost = getCost(labels, predictions)
    accuracy = getAccuracy(labels, predictions)
       
    print(cost,predictions)
    optimiser = getOptimiser(lr, cost)
    
    saver = tf.train.Saver()

    for var in tf.trainable_variables('fully_connected'):
        name = 'fc-{0}'
        if 'weights' in var.name:
            name = name.format('weights')
        else:
            name = name.format('biases')
        with tf.variable_scope(name):
            formatSummaries(var)  
    
    for var in tf.trainable_variables('dense'):
        print(var.name)
        name = var.name.split("/")[0]+"-{0}"
        if 'kernel' in var.name:
            name = name.format('weights')
        else:
            name = name.format('biases')
        with tf.variable_scope(name):
            formatSummaries(var)  

    merged = tf.summary.merge_all()


# ### Training of the network

# In[ ]:


import time
start_time = time.time()

with tf.device('/gpu:0'):
    with tf.Session(graph=train_graph) as sess:
        train_writer = tf.summary.FileWriter("./tensorboard-logs/try{0}/train".format(experimentNumber),
                                      sess.graph)
        validation_writer = tf.summary.FileWriter("./tensorboard-logs/try{0}/val".format(experimentNumber),
                              sess.graph)

        sess.run(tf.global_variables_initializer())
        iteration = 1
        for epoch in range(epochNumber):
            for index, (x, y) in enumerate(getBatches(train_x, train_y, batchSize),1):
                feed = {
                        inputs: x, 
                        labels:y, 
                        lr: learningRate,
                        drop_keep_porob: keepProbability}
                
                sess.run([optimiser], feed_dict=feed)

                if iteration%5==0:
                    summary,loss = sess.run([merged, cost],
                        feed_dict=feed)                    
                    print(
                    "Epoch: {0}/{1}".format(epoch, epochNumber),
                    "Iteration: {0}".format(iteration),
                    "Train loss: {:.8f}".format(loss))
                    train_writer.add_summary(summary, iteration)
                    
                if iteration%25==0:
                    val_acc = []
                    for x1,y1 in getBatches(val_x, val_y, batchSize):
                        feed = {inputs: x1, 
                                labels: y1,
                                lr: learningRate,
                                drop_keep_porob:1.0}
                        summary,batch_acc = sess.run([merged, accuracy], feed_dict=feed)
                        val_acc.append(batch_acc)
                    print("Val acc: {:.8f}".format(np.mean(val_acc)))
                    validation_writer.add_summary(summary, iteration)

                iteration +=1
            saver.save(sess, "checkpoints/model{0}/predictor.ckpt".format(modelNumber))

    print("--- %s seconds ---" % (time.time() - start_time))


# In[19]:


test_acc = []
with tf.Session(graph=train_graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints/model{0}'.format(modelNumber)))
    for ii, (x, y) in enumerate(getBatches(test_x, test_y, batchSize), 1):
        feed = {inputs: x,
                labels: y,
                drop_keep_porob: 1.}
        batch_acc = sess.run([accuracy], feed_dict=feed)
        test_acc.append(batch_acc)
    print("Test accuracy: {:.3f}".format(1-np.mean(test_acc)))

