# -*- coding: utf-8 -*-
"""
Created on Wed Apr 04 13:32:40 2018

@author: S06077
"""

#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
import nltk
import string
import join
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
from sklearn.metrics import accuracy_score

#Load Data
train = pd.read_csv("D:/.../input/train_E6oV3lV.csv", encoding='ISO-8859-1')
test = pd.read_csv("D:/.../input/test_tweets_anuFYb8.csv", encoding='ISO-8859-1')

#Function Clean Data
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
stop_words = stopwords.words('english')
tok = WordPunctTokenizer()
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

#Clean Train Data Tweet
training = train.tweet
dftext = []
for t in training:
    dftext.append(tweet_cleaner(t))

#Transform column label to list
dfList = train['label'].tolist()

#Transformation to dataframe
dataset = sklearn.datasets.base.Bunch(data=dftext, target=dfList)
train = pd.DataFrame(dataset)
train.columns = ['tweet','label']


#Build Pipeline
#MultinomialNB
pipeline = Pipeline([
    ('count_vectorizer',   CountVectorizer()),
    ('tfidf_transformer',  TfidfTransformer(use_idf = True)),
    ('classifier',         MultinomialNB(alpha = 0.01))
])

#SGDClassifier
pipeline = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,)),
])

#Tune Model
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])
parameters = {  
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'), 
    'classifier__fit_prior': (True, False),  
    'classifier__class_prior': [0.1, 0.9],  
    'classifier__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
    } 

pipeline = Pipeline([
    ('vect',   CountVectorizer()),
    ('tfidf',  TfidfTransformer()),
    ('clf',  MultinomialNB())
])
parameters = {  
    'vect__max_df': (0.5, 0.625, 0.75, 0.875, 1.0),  
    'vect__max_features': (None, 5000, 10000, 20000),  
    'vect__min_df': (1, 5, 10, 20, 50),  
    'tfidf__use_idf': (True, False),  
    'tfidf__sublinear_tf': (True, False),  
    'vect__binary': (True, False),  
    'tfidf__norm': ('l1', 'l2'),  
    'clf__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001)  
    } 

#Find Best Parameters
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
print(parameters)
#t0 = time()
grid_search.fit(train.tweet, train.label)

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

    
#Make prediction with Train Data                    
k_fold = KFold(n=len(train), n_folds=6)
scores = []
confusion = np.array([[0, 0], [0, 0]])
for train_indices, test_indices in k_fold:
    train_text = train.iloc[train_indices]['tweet'].values
    train_y = train.iloc[train_indices]['label'].values

    test_text = train.iloc[test_indices]['tweet'].values
    test_y = train.iloc[test_indices]['label'].values

    pipeline.fit(train_text, train_y)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_y, predictions)
    score = f1_score(test_y, predictions, average= 'binary')
    scores.append(score)

print('Total tweets classified:', len(train))
print('Score:', sum(scores)/len(scores))
print('Confusion matrix:')
print(confusion)

#Clean Test Data
testing = test.tweet
dftest = []
for t in testing:
    dftest.append(tweet_cleaner(t))
    
#Make Predictions
predictions = pipeline.predict(dftest)
predictions = pd.DataFrame(predictions)

#Download Predictions
frames = [test, predictions]
submission = pd.concat(frames, axis=1, join_axes=[test.index])
submission = submission[['id',0]]
submission.columns = ['id', 'label']
submission = submission[['id', 'label']]
submission.to_csv('submission.csv')
