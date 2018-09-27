import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def Plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8,
                    c=cmap(idx), marker=markers[idx], label=cl)


class AdalineGD(object):
    '''
    ADAptive Linear Neuron classifier.

    Parameters:
    eta: float
        Learning rate(between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.

    Attributes:
    w_: id-array.
        Weights after fitting.
    
    errors: list
        Number of misclassifications in every epoch.
    '''

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y):
        '''
        Fit training data

        Parameters
        ---------
        x: (Array-like),shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and 
            n_features is the number of features.

        y : array-like, Shape = [n_samples]
            Target values.

        Returns
        ------
        self : object
        '''
        self.w_ = np.zeros(1+x.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            output = self.net_input(x)
            errors = (y - output)
            #update weight
            self.w_[1:] += self.eta * x.T.dot(errors)
            # self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        '''
        Calculate net input
        '''
        return np.dot(x, self.w_[1:] + self.w_[0])

    def activation(self, x):
        '''Compute linear activation'''
        return self.net_input(x)

    def predict(self, x):
        '''Return class label after unit step'''
        return np.where(self.activation(x) >= 0.0, 1, -1)


#inport iris.data
df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

#process iris data
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values

'''
show iris data
'''
# plt.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(x[50:100, 0], x[50:100, 1], color='blue',
#             marker='x', label='versicolor')
# plt.xlabel('petal length')
# plt.ylabel('sepal length')
# plt.legend(loc='upper left')
# plt.show()

#show AdalineGD reslut graphic with two different rate(0.01,0.0001)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

#define iter count: 10 
# ada1 = AdalineGD(n_iter=10, eta=0.01).fit(x, y)
# ax[0].plot(range(1, len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
# ax[0].set_xlabel('Epochs')
# ax[0].set_ylabel('log(Sum-squared-error)')
# ax[0].set_title('Adaline - learning rate 0.01')

# ada2 = AdalineGD(n_iter=50, eta=0.0001).fit(x, y)
# ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker='o')
# ax[1].set_xlabel('Epochs')
# ax[1].set_ylabel('Sum-squared-error')
# ax[1].set_title('Adaline - learning rate 0.0001')
# plt.show()

#standardize process: 符合正太分布
#x[].std(): 标准差
#x[].mean(): 平均值
x_std = np.copy(x)
x_std[:,0] = (x[:,0] - x[:,0].mean()) / x[:,0].std()
x_std[:,1] = (x[:,1] - x[:,1].mean()) / x[:,1].std()

#It will be convergence after standard process with 15 count.
ada = AdalineGD(n_iter=15,eta=0.01)
ada.fit(x_std,y)

#show region graphic
# Plot_decision_regions(x_std,y,classifier=ada)
# plt.title('Adaline - Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('sepal length [standardized]')
# plt.legend(loc='upper left')
# plt.show()

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()
