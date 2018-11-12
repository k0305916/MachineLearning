import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed


class AdalineSGD(object):
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

    shuffle: bool(default: True)
        Shuffles training data every epoch
        if True to prevent cycles
    
    random_state: int(default: None)
        Set random state for shuffling and initializing the weights
    '''

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

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
        self._initialize_weights(x.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
        return self

    #用于处理流数据的在线学习
    def parial_fit(self, x, y):
        '''
        Fit training data without weinitializing the weights
        '''
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self

    def _shuffle(self, x, y):
        '''Shuffle training data'''
        #permutation(int i): i is range(0~i-1)
        r = np.random.permutation(len(y))
        return x[r], y[r]

    def _initialize_weights(self, m):
        '''Initialize weights to zeros'''
        self.w_ = np.zeros(1+m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        '''Apply Adaline learning rule to update the weights'''
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        # self.w_[0] += self.eta * error
        cost = 0.5*error**2
        return cost

    def net_input(self, x):
        '''calculate net input'''
        return np.dot(x, self.w_[1:])+self.w_[0]

    def activation(self, x):
        '''Compute linear activation'''
        return self.net_input(x)

    def predict(self, x):
        '''Return class label after unit step'''
        return np.where(self.activation(x) >= 0.0, 1, -1)


def plot_decision_regions(x, y, classifier, resolution=0.02):
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


df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
x = df.iloc[0:100, [0, 2]].values

x_std = np.copy(x)
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()
x_std[:, 1] = (x[:, 1] - x[:, 1].mean()) / x[:, 1].std()

ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(x_std, y)

plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

# plot_decision_regions(x_std, y, classifier=ada)
# plt.title('Adaline - Stochastic Gradient Descent')
# plt.xlabel('Sepal length [Standardized]')
# plt.ylabel('Sepal length [Standardized]')
# plt.legend(loc='upper left')
# plt.show()
