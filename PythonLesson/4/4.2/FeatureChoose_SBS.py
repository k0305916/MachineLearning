from sklearn.base import clone
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SBS():
    def __init__(self,estimator,k_feature,scoring=accuracy_score,
        test_size=0.25,random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_feature = k_feature
        self.test_size = test_size
        self.random_state = random_state

    def fit(self,x,y):
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=self.test_size,
            random_state=self.random_state)
        dim = x_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(x_train,y_train,x_test,y_test,self.indices_)
        self.scores_ = [score]
        while dim > self.k_feature:
            scores = []
            subsets = []

            for p in combinations(self.indices_,r=dim-1):
                score = self._calc_score(x_train,y_train,x_test,y_test,p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self,x):
        return x[:,self.indices_]

    def _calc_score(self,x_train,y_train,x_test,y_test,indices):
        self.estimator.fit(x_train[:,indices],y_train)
        y_pred = self.estimator.predict(x_test[:,indices])
        score = self.scoring(y_test,y_pred)

        return score

df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# standardScaler
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn,k_feature=1)
sbs.fit(x_train_std,y_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat,sbs.scores_,marker='o')
plt.ylim([0.7,1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of deatures')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

knn.fit(x_train_std,y_train)
print('Training accuracy:', knn.score(x_train_std,y_train))
print('Test accuracy:', knn.score(x_test_std,y_test))

knn.fit(x_train_std[:,k5],y_train)
print('Training accuracy:', knn.score(x_train_std[:,k5],y_train))
print('Test accuracy:', knn.score(x_test_std[:,k5],y_test))



