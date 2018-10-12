#<a tutorial on support vector machines for pattern recognition>
#Explain 二次规划

#???what is the SGDClassifier?
#sklearn :: SGDClassifier(loss='perceptron/log/hinge')

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plot_decision_regions(x, y, classifier, test_idx=None, resultion=0.02):
    #setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resultion),
                           np.arange(x2_min, x2_max, resultion))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.xlim(xx2.min(), xx2.max())

    #plot all samples
    x_test, y_test = x[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    #highlight test samples
    if test_idx:
        x_test, y_test = x[test_idx, :], y[test_idx]
        plt.scatter(x_test[:, 0], x_test[:, 1], c='y',
                    alpha=0.8, linewidth=1, marker='o',
                    s=55, label='test set')

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
x_test_std = sc.transform(x_test)

svm = SVC(kernel='linear',C=1.0,random_state=0)
svm.fit(x_train_std,y_train)
x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))


plot_decision_regions(x_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
