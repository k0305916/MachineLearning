# 如果要在不同机器学习算法中做出选择，则推荐另外一种方法——嵌套交叉验证，

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
x = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)
# le.transform(['M','B'])


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np

pipe_svc = Pipeline([('scl',StandardScaler()),
                ('clf',SVC(random_state=1))])
param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid=[{'clf__C':param_range,
            'clf__kernel':['linear']},
            {'clf__C':param_range,
            'clf__gamma':param_range,
            'clf__kernel':['rbf']}]


gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)

from sklearn.model_selection import cross_val_score
scores=cross_val_score(gs,x,y,scoring='accuracy',cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))


# DicisionTree
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),param_grid=[{'max_depth':[1,2,3,4,5,6,7,None]}],
    scoring='accuracy',cv=5)
scores=cross_val_score(gs,x_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores),np.std(scores)))    