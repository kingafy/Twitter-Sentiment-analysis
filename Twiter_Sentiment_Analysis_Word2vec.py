# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 10:26:30 2018

@author: Anshuman_Mahapatra
"""

##Sentiment Analysis using Word2VEc

import pandas as pd # provide sql-like data manipulation tools. very handy.
pd.options.mode.chained_assignment = None
import numpy as np # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



###Data Ingestion####

def ingest():
    data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/train_tweets.csv",encoding="utf-8")
    data.drop(['id'], axis=1, inplace=True)
    data = data[data.label.isnull() == False]
    data['label'] = data['label'].map(int)
    data = data[data['tweet'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print('dataset loaded with shape', data.shape)    
    return data

data = ingest()
data.head(5)

##Clean section splits each tweet into tokens and removes user mentions, hashtags and urls.

def tokenize(tweet1):
    try:
        tweet1 = tweet1.lower()
        tokens = tokenizer.tokenize(tweet1)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'
    
 ####Tokenizing the tweets   
def postprocess(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['tweet'].progress_map(tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data = postprocess(data)

print(data.head())


##Building Word2vec model

x_train, x_test, y_train, y_test = train_test_split(np.array(data.tokens),np.array(data.label), test_size=0.2)



print(x_train.shape)


##turn them into LabeledSentence objects
def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

##print(x_train)

##Word2Vec creation
n_dim = 200
tweet_w2v = Word2Vec(size=n_dim, min_count=10)
tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count, epochs=tweet_w2v.iter)


tweet_w2v['good']

tweet_w2v.most_similar('good')

tweet_w2v.most_similar('bar')

###Visualisations
# importing bokeh library for interactive dataviz
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

# defining the chart

'''
output_notebook()
plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="A map of 10000 word vectors",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

# getting a list of word vectors. limit to 10000. each is of 200 dimensions
word_vectors = [tweet_w2v[w] for w in list(tweet_w2v.wv.vocab.keys())[:10000]]

##print(word_vectors)

# dimensionality reduction. converting the vectors to 2d vectors for visualization
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
tsne_w2v = tsne_model.fit_transform(word_vectors)

# putting everything in a dataframe
tsne_df = pd.DataFrame(tsne_w2v, columns=['x', 'y'])
tsne_df['words'] = list(tweet_w2v.wv.vocab.keys())[:10000]

# plotting. the corresponding word appears when you hover on the data point.
from bokeh.models import ColumnDataSource
plot_tfidf.scatter(x='x', y='y', source=ColumnDataSource(tsne_df))
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"word": "@words"}
show(plot_tfidf)
'''


print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


from sklearn.preprocessing import scale
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=200))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_vecs_w2v, y_train, epochs=9, batch_size=32, verbose=2)

score = model.evaluate(test_vecs_w2v, y_test, batch_size=128, verbose=2)
print(score[1])



###Predicting the Test Data set provided
def ingest():
    data = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv",encoding="utf-8")
    data.drop(['id'], axis=1, inplace=True)
    data = data[data['tweet'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print('dataset loaded with shape', data.shape)    
    return data

data_test= ingest()


data_test.head(5)

def tokenize_test(tweet1):
    try:
        tweet1 = tweet1.lower()
        tokens = tokenizer.tokenize(tweet1)
        tokens = list(filter(lambda t: not t.startswith('@'), tokens))
        tokens = list(filter(lambda t: not t.startswith('#'), tokens))
        tokens = list(filter(lambda t: not t.startswith('http'), tokens))
        return tokens
    except:
        return 'NC'
    
 ####Tokenizing the tweets   
def postprocess_test(data, n=1000000):
    data = data.head(n)
    data['tokens'] = data['tweet'].progress_map(tokenize_test)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data

data_test = postprocess_test(data_test)

print(data_test)
data_test_tok =np.array(data_test.tokens)


def labelizeTweets(tweets, label_type):
    labelized = []
    for i,v in tqdm(enumerate(tweets)):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

data_test = labelizeTweets(data_test_tok, 'FINAL')

print(data_test)

test_vecs_w2v_final = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, data_test))])
test_vecs_w2v_final = scale(test_vecs_w2v_final)

predict_final_data = model.predict_classes(test_vecs_w2v_final)

print(predict_final_data)


test_df = pd.read_csv("D:/Data Science/POC/Twitter Sentimental Analysis/test_tweets.csv")

test_df['label'] = predict_final_data.reshape(-1,1)
test_df.to_csv("D:/Data Science/POC/Twitter Sentimental Analysis/Word2Vec_pred.csv",index=False)