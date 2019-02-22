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

final_result = pd.DataFrame({'id':test_data['id'],'label':predictions})
final_result.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/output_countvectors.csv",index=False)


##runnig Grid search for Randomforest
# build a classifier
##clf = RandomForestClassifier(n_estimators=20)

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
param_grid = {"n_estimators" :[200],
              "max_depth": [3, None],
              "max_features": [50, 100],
              "min_samples_split": [3, 8],
              "min_samples_leaf": [3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run grid search
grid_search_rf = GridSearchCV(clf, param_grid=param_grid)
grid_search_rf.fit(x_train_tfidf,y_train)

'''
{'bootstrap': False, 'criterion': 'entropy', 'max_depth': None, 'max_features': 100, 'min_samples_leaf': 3, 'min_samples_split': 3, 'n_estimators': 200}
'''


model_tuned  = RandomForestClassifier(n_estimators = 200, criterion = "entropy", max_features = 100, min_samples_leaf =3, min_samples_split = 3)

model_tuned.fit(x_train_tfidf,y_train)

predictions_tuned = model_tuned.predict(x_test_tfidf)


from sklearn.metrics import confusion_matrix,f1_score
confusion_matrix(y_test,predictions_tuned)
print(confusion_matrix(y_test,predictions_tuned))

f1_score(y_test,predictions_tuned)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions_tuned))

###Run for Test Data

test_data1 = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv")

test_data1['processed_tweet'] = test_data1['tweet'].apply(process_tweet)


## for transforming the whole train data ##
train_counts = count_vect.fit_transform(train_data['processed_tweets'])
train_tfidf = transformer.fit_transform(train_counts)
## for transforming the test data ##
test_counts = count_vect.transform(test_data1['processed_tweet'])
test_tfidf = transformer.transform(test_counts)
## fitting the model on the transformed train data ##
model_tuned.fit(train_tfidf,train_data['label'])
## predicting the results ##
predictions_tuned1 = model.predict(test_tfidf)

final_result = pd.DataFrame({'id':test_data['id'],'label':predictions_tuned1})
final_result.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/output_tuned_countvectors.csv",index=False)