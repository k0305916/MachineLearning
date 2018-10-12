from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

#load iris data
iris = datasets.load_iris()
x = iris.data[:, [2, 3]]
#iris.target can change class to 0,1,2... automaticly.
y = iris.target

#split train and test data.
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

#standard data: std;
sc = StandardScaler()
#sc.fit can account 平均值 and 方差.
sc.fit(x_train)
x_train_std = sc.transform(x_train)


weights,params = [],[]
for c in np.arange(-5,5,dtype=float):
    lr = LogisticRegression(C=10**c,random_state=0)
    lr.fit(x_train_std,y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)

plt.plot(params,weights[:,0],label='petal length')
plt.plot(params,weights[:,1],linestyle='--',label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()