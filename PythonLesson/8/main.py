import pyprind
import pandas as pd
import os

from nltk.stem.porter import PorterStemmer
def tokenizer(text):
    return text.split()

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

# 首先初始化了一个包含50000次迭代的进度条对象pbar
pbar = pyprind.ProgBar(50000)
labels = {'pos':1,'neg':0}
df = pd.DataFrame()

for s in ('test','train'):
    for l in ('pos','neg'):
        path='E:\\Business\\Python\\aclImdb\\%s\\%s' % (s,l)
        for file in os.listdir(path):
            with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt,labels[l]]],ignore_index=True)
            pbar.update()
df.columns = ['review','sentiment']

import numpy as np
np.random.seed(0)
df = df.reindex(np.random.pernutation(df.index))
df.to_csv('./movie_data.csv',index=False)

# comfirm the data right or wrong.---It is not necessary
df = pd.read_csv('./movie_data.csv')
df.head(3)

# split train and test data
x_train = df.loc[:25000,'review'].values
y_train = df.loc[:25000,'sentiment'].values
x_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values

# get the best performance parameters through GridSearch
form sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
form sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid=[{'vect__ngram_range':[(1,1)],
        'vect__stop_words':[stop,None],
        'vect__tokenizer':[tokenizer,tokenizer_porter],
        'clf__penalty':['l1','l2'],
        'clf__C':[1.0,10.0,100.0]},
        {'vect__ngram_range':[(1,1)],
        'vect__stop_words':[stop,None],
        'vect__tokenizer':[tokenizer,tokenizer_porter],
        'vect__use_idf':[False],
        'vect__norm':[None],
        'clf__penalty':['l1','l2'],
        'clf__C':[1.0,10.0,100.0]}
        ]
lr_tfidf = Pipeline([('vect',tfidf),
                    ('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)                    
gs_lr_tfidf.fit(x_train,y_train)

print('Best parameter set: %s'%gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f'%gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f'%clf.score(x_test,y_test))



