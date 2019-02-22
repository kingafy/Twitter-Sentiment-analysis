# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:47:22 2018

@author: Anshuman_Mahapatra
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:29:58 2018

@author: Anshuman_Mahapatra
"""

##Twitter Sentiment Analysis ++Boost it

import numpy as np
import pandas as pd


import seaborn as sns
import matplotlib.pyplot as plt

import math
import random
from collections import defaultdict
from pprint import pprint

# Prevent future/deprecation warnings from showing in output
import warnings
warnings.filterwarnings(action='ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16,9)})

train_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/train_tweets.csv")
train_data.head()
######SPARSE MATRIX METHOD
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=1000, binary=True)
X = vect.fit_transform(train_data.tweet)

X.toarray()

####DATA SPLIT

from sklearn.model_selection import train_test_split

X = train_data.tweet
y = train_data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##Creating vector based on Training Data only to have some unexpected words in test

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000, binary=True)

X_train_vect = vect.fit_transform(X_train)

##BALANCING THE DATA

counts = train_data.label.value_counts()
print(counts)

print("\nPredicting only 0 = {:.2f}% accuracy".format(counts[0] / sum(counts) * 100))

##by default system is predicting 93% as 0 which is quite high in terms of accuracy

##performing SMOTE to do Data balancing

from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)


unique, counts = np.unique(y_train_res, return_counts=True)
print(list(zip(unique, counts)))


###NAIVE BAYES######
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)

nb.score(X_train_res, y_train_res)

##Accuracyon Train Data or overall fit accuracy:- 0.8903913992944734

##Convert Test Data into Test vector

X_test_vect = vect.transform(X_test)

y_pred = nb.predict(X_test_vect)

y_pred


###Accuracy metrics with Test data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
'''
Accuracy: 85.28%

F1 Score: 44.81

COnfusion Matrix:
 [[5070  838]
 [ 103  382]]
 '''
 
 
##Prediction of test Data set
 
av_test_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv") 

av_test_vect = vect.transform(av_test_data.tweet)
av_test_vect.toarray()
y_pred_nb = nb.predict(av_test_vect)

y_pred_nb


##wrtie to final file
final_result = pd.DataFrame({'id':av_test_data['id'],'label':y_pred_nb})
final_result.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/nb.csv",index=False)
####Cross fold on NB#############################
 
from sklearn.model_selection import ShuffleSplit

X = train_data.tweet
y = train_data.label

ss = ShuffleSplit(n_splits=10, test_size=0.2)
sm = SMOTE()

accs = []
f1s = []
cms = []

for train_index, test_index in ss.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit vectorizer and transform X train, then transform X test
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)
    
    # Oversample
    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
    
    # Fit Naive Bayes on the vectorized X with y train labels, 
    # then predict new y labels using X test
    nb.fit(X_train_res, y_train_res)
    y_pred = nb.predict(X_test_vect)
    
    # Determine test set accuracy and f1 score on this fold using the true y labels and predicted y labels
    accs.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    cms.append(confusion_matrix(y_test, y_pred))
    
print("\nAverage accuracy across folds: {:.2f}%".format(sum(accs) / len(accs) * 100))
print("\nAverage F1 score across folds: {:.2f}%".format(sum(f1s) / len(f1s) * 100))
print("\nAverage Confusion Matrix across folds: \n {}".format(sum(cms) / len(cms)))

'''
Average accuracy across folds: 86.04%

Average F1 score across folds: 45.66%

Average Confusion Matrix across folds: 
 [[5125.4  808.2]
 [  84.3  375.1]]
 '''
###Kfold for Test data
##av_test_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv") 

av_test_vect_kfold = vect.transform(av_test_data.tweet)
av_test_vect_kfold.toarray()
y_pred_nb_kfold = nb.predict(av_test_vect_kfold)

y_pred_nb_kfold


##wrtie to final file
final_result_kfold = pd.DataFrame({'id':av_test_data['id'],'label':y_pred_nb_kfold})
final_result_kfold.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/nb_kfold.csv",index=False)



#######Do data cleaning and check if model boosts##################33
## importing regular expression library ##
import re
def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", "",tweet.lower()).split())

train_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/train_tweets.csv")
train_data.head()
train_data['tweet'] = train_data['tweet'].apply(process_tweet)


######SPARSE MATRIX METHOD
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=1000, binary=True)
X = vect.fit_transform(train_data.tweet)

X.toarray()

####DATA SPLIT

from sklearn.model_selection import train_test_split

X = train_data.tweet
y = train_data.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

##Creating vector based on Training Data only to have some unexpected words in test

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000, binary=True)

X_train_vect = vect.fit_transform(X_train)

##BALANCING THE DATA

counts = train_data.label.value_counts()
print(counts)

print("\nPredicting only 0 = {:.2f}% accuracy".format(counts[0] / sum(counts) * 100))

##by default system is predicting 93% as 0 which is quite high in terms of accuracy

##performing SMOTE to do Data balancing

from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)


unique, counts = np.unique(y_train_res, return_counts=True)
print(list(zip(unique, counts)))


###NAIVE BAYES######
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)

nb.score(X_train_res, y_train_res)

##Accuracyon Train Data or overall fit accuracy:- 0.8903913992944734

##Convert Test Data into Test vector

X_test_vect = vect.transform(X_test)

y_pred = nb.predict(X_test_vect)

y_pred


###Accuracy metrics with Test data
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))

'''
Accuracy: 86.03%

F1 Score: 43.59

COnfusion Matrix:
 [[5155  799]
 [  94  345]]
 '''
 
 ##Prediction of test Data set
 
av_test_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv") 
av_test_data['tweet'] = av_test_data['tweet'].apply(process_tweet)
av_test_vect = vect.transform(av_test_data.tweet)
av_test_vect.toarray()
y_pred_nbc = nb.predict(av_test_vect)

y_pred_nbc


##wrtie to final file
final_result = pd.DataFrame({'id':av_test_data['id'],'label':y_pred_nbc})
final_result.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/nbc.csv",index=False)


##############################KFOLD after data cleaning##########
from sklearn.model_selection import ShuffleSplit

train_data.head()

##Train and test date is already cleaned in above so no need to apply it again here
X = train_data.tweet
y = train_data.label

ss = ShuffleSplit(n_splits=10, test_size=0.2)
sm = SMOTE()

accs = []
f1s = []
cms = []

for train_index, test_index in ss.split(X):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Fit vectorizer and transform X train, then transform X test
    X_train_vect = vect.fit_transform(X_train)
    X_test_vect = vect.transform(X_test)
    
    # Oversample
    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
    
    # Fit Naive Bayes on the vectorized X with y train labels, 
    # then predict new y labels using X test
    nb.fit(X_train_res, y_train_res)
    y_pred = nb.predict(X_test_vect)
    
    # Determine test set accuracy and f1 score on this fold using the true y labels and predicted y labels
    accs.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    cms.append(confusion_matrix(y_test, y_pred))
    
print("\nAverage accuracy across folds: {:.2f}%".format(sum(accs) / len(accs) * 100))
print("\nAverage F1 score across folds: {:.2f}%".format(sum(f1s) / len(f1s) * 100))
print("\nAverage Confusion Matrix across folds: \n {}".format(sum(cms) / len(cms)))

'''
Average accuracy across folds: 86.01%

Average F1 score across folds: 43.99%

Average Confusion Matrix across folds: 
 [[5147.   800.4]
 [  94.   351.6]]
 '''
###Kfold for Test data
##av_test_data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv") 

av_test_vect_kfold = vect.transform(av_test_data.tweet)
av_test_vect_kfold.toarray()
y_pred_nbc_kfold = nb.predict(av_test_vect_kfold)

y_pred_nbc_kfold


##wrtie to final file
final_result_kfold = pd.DataFrame({'id':av_test_data['id'],'label':y_pred_nbc_kfold})
final_result_kfold.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/nbc_kfold.csv",index=False)



#####TRYING MIX OF ALL MODELS
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

X = train_data.tweet
y = train_data.label

cv = ShuffleSplit(n_splits=20, test_size=0.2)

models = [
    MultinomialNB(),
    BernoulliNB(),
    LogisticRegression(),
    SGDClassifier(),
    LinearSVC(),
    RandomForestClassifier(),
    MLPClassifier()
]

sm = SMOTE()

# Init a dictionary for storing results of each run for each model
results = {
    model.__class__.__name__: {
        'accuracy': [], 
        'f1_score': [],
        'confusion_matrix': []
    } for model in models
}

for train_index, test_index in cv.split(X):
    X_train, X_test  = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_vect = vect.fit_transform(X_train)    
    X_test_vect = vect.transform(X_test)
    
    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
    
    for model in models:
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_vect)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[model.__class__.__name__]['accuracy'].append(acc)
        results[model.__class__.__name__]['f1_score'].append(f1)
        results[model.__class__.__name__]['confusion_matrix'].append(cm)
        
for model, d in results.items():
    avg_acc = sum(d['accuracy']) / len(d['accuracy']) * 100
    avg_f1 = sum(d['f1_score']) / len(d['f1_score']) * 100
    avg_cm = sum(d['confusion_matrix']) / len(d['confusion_matrix'])
    
    slashes = '-' * 30
    
    s = f"""{model}\n{slashes}
        Avg. Accuracy: {avg_acc:.2f}%
        Avg. F1 Score: {avg_f1:.2f}
        Avg. Confusion Matrix: 
        \n{avg_cm}
        """
    print(s)
    
'''    
MultinomialNB
------------------------------
        Avg. Accuracy: 86.23%
        Avg. F1 Score: 44.72
        Avg. Confusion Matrix: 
        
[[5156.05  789.2 ]
 [  91.35  356.4 ]]
        
BernoulliNB
------------------------------
        Avg. Accuracy: 84.59%
        Avg. F1 Score: 42.70
        Avg. Confusion Matrix: 
        
[[5040.15  905.1 ]
 [  80.35  367.4 ]]
        
LogisticRegression
------------------------------
        Avg. Accuracy: 85.25%
        Avg. F1 Score: 42.37
        Avg. Confusion Matrix: 
        
[[5103.15  842.1 ]
 [ 100.95  346.8 ]]
        
SGDClassifier
------------------------------
        Avg. Accuracy: 83.76%
        Avg. F1 Score: 40.54
        Avg. Confusion Matrix: 
        
[[5001.6   943.65]
 [  94.3   353.45]]
        
LinearSVC
------------------------------
        Avg. Accuracy: 84.87%
        Avg. F1 Score: 41.55
        Avg. Confusion Matrix: 
        
[[5082.25  863.  ]
 [ 103.95  343.8 ]]
        
RandomForestClassifier
------------------------------
        Avg. Accuracy: 94.75%
        Avg. F1 Score: 53.58
        Avg. Confusion Matrix: 
        
[[5863.2    82.05]
 [ 253.65  194.1 ]]
        
MLPClassifier
------------------------------
        Avg. Accuracy: 93.36%
        Avg. F1 Score: 55.76
        Avg. Confusion Matrix: 
        
[[5701.55  243.7 ]
 [ 180.8   266.95]]
'''    


####Ensembling

from sklearn.ensemble import VotingClassifier

X = train_data.tweet
y = train_data.label

cv = ShuffleSplit(n_splits=10, test_size=0.2)

models = [
    MultinomialNB(),
    BernoulliNB(),
    LogisticRegression(),
    SGDClassifier(),
    LinearSVC(),
    RandomForestClassifier(),
    MLPClassifier()
]

m_names = [m.__class__.__name__ for m in models]

models = list(zip(m_names, models))
vc = VotingClassifier(estimators=models)

sm = SMOTE()

# No need for dictionary now
accs = []
f1s = []
cms = []

for train_index, test_index in cv.split(X):
    X_train, X_test  = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    X_train_vect = vect.fit_transform(X_train)    
    X_test_vect = vect.transform(X_test)
    
    X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
    
    vc.fit(X_train_res, y_train_res)
    
    y_pred = vc.predict(X_test_vect)
    
    accs.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred))
    cms.append(confusion_matrix(y_test, y_pred))    
    
    
print("Voting Classifier")
print("-" * 30)
print("Avg. Accuracy: {:.2f}%".format(sum(accs) / len(accs) * 100))
print("Avg. F1 Score: {:.2f}".format(sum(f1s) / len(f1s) * 100))
print("Confusion Matrix:\n", sum(cms) / len(cms))
'''
Voting Classifier
------------------------------
Avg. Accuracy: 88.60%
Avg. F1 Score: 47.75
Confusion Matrix:
 [[5331.7  619.1]
 [ 109.5  332.7]]
''''
#################################################

y_pred_enc = vc.predict(av_test_vect_kfold)

y_pred_enc


##wrtie to final file
final_result_enc = pd.DataFrame({'id':av_test_data['id'],'label':y_pred_enc})
final_result_enc.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/enc.csv",index=False)

 