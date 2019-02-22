# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:29:58 2018

@author: Anshuman_Mahapatra
"""

##Twitter Sentiment Analysis

import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt



train_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/train_tweets.csv")
train_data.head()


##EDA
train_data.info()

##Check the  count of 1 and 0

train_data['label'].value_counts()

train_data['tweet'].head()

'''
0     @user when a father is dysfunctional and is s...
1    @user @user thanks for #lyft credit i can't us...
2                                  bihday your majesty
3    #model   i love u take with u all the time in ...
4               factsguide: society now    #motivation
'''
##Data cleaning

## importing regular expression library ##
import re
def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "",tweet.lower()).split())


train_data['processed_tweets'] = train_data['tweet'].apply(process_tweet)

train_data.head(10)


###Drop features which are not required
def drop_features(features,data):
    data.drop(features,inplace=True,axis=1)
    

drop_features(['id','tweet'],train_data)


##check the new definiton
train_data.head()

###Data split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_data["processed_tweets"],train_data["label"], test_size = 0.2, random_state = 42)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)


print(x_train_counts.shape)
print(x_train_tfidf.shape)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)


print(x_test_counts.shape)
print(x_test_tfidf.shape)



from  sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200)
model.fit(x_train_tfidf,y_train)

predictions = model.predict(x_test_tfidf)

from sklearn.metrics import confusion_matrix,f1_score
confusion_matrix(y_test,predictions)
print(confusion_matrix(y_test,predictions))

f1_score(y_test,predictions)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))


###Run for Test Data

test_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv")

test_data['processed_tweet'] = test_data['tweet'].apply(process_tweet)


## for transforming the whole train data ##
train_counts = count_vect.fit_transform(train_data['processed_tweets'])
train_tfidf = transformer.fit_transform(train_counts)
## for transforming the test data ##
test_counts = count_vect.transform(test_data['processed_tweet'])
test_tfidf = transformer.transform(test_counts)
## fitting the model on the transformed train data ##
model.fit(train_tfidf,train_data['label'])
## predicting the results ##
predictions = model.predict(test_tfidf)

print(len(predictions))

final_result = pd.DataFrame({'id':test_data['id'],'label':predictions})
final_result.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/output_countvectors.csv",index=False)

######APPLY DEEP LEARNING#######################
import numpy as np 
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re


max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_data["processed_tweets"].values)
X = tokenizer.texts_to_sequences(train_data["processed_tweets"].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(train_data["label"]).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

##Model training
batch_size = 1000
model.fit(X_train, Y_train, epochs = 30, batch_size=batch_size, verbose = 2)

model.save("D:/Data Science/POC/Twitter Sentimental Analysis/LSTM_sentiment_ep50.h5")


##prediction phase


X_test = tokenizer.texts_to_sequences(test_data["processed_tweet"].values)
X_test = pad_sequences(X_test,maxlen=32, dtype='int32', value=0)
#padding the tweet to have exactly the same shape as `embedding_2` input
print(X_test)
print(len(X_test))

print(X_test.shape)


test_pred = model.predict_classes(X_test)

# edits the test file to input the prediction labels
test_df = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv")

test_df['label'] = test_pred.reshape(-1,1) 

test_df.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/final_predicted_sentiments.csv",index=False)
'''
test_tweets_predict  = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv")

final_pred = np.array([])
for i in range(0,len(test_tweets_predict)):
    final_pred = model.predict(X_test[i],batch_size=1,verbose = 2)[0]
    test_tweets_predict.label[i] = final_pred

sentiment = model.predict(X_test,batch_size=1,verbose = 2)
print(len(sentiment))
print(sentiment)
'''

sentiment.DataFrame.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/output_LSTM.csv",index=False)
'''
final_result1 = pd.DataFrame({'id':test_data['id'],'label':sentiment})
final_result1.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/output_LSTM.csv",index=False)
'''
