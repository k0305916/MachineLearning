from sklearn.svm import SVC
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

np.random.seed(0)
x_xor = np.random.randn(200,2)
y_xor = np.logical_xor(x_xor[:,0] > 0,x_xor[:,1] > 0)
y_xor = np.where(y_xor,1,-1)

# plt.scatter(x_xor[y_xor==1,0],x_xor[y_xor==1,1],c='b',marker='s',label='1')
# plt.scatter(x_xor[y_xor==-1,0],x_xor[y_xor==-1,1],c='r',marker='x',label='-1')
# plt.ylim(-3.0)
# plt.legend()
# plt.show()

#reference rbf kernel function---the most point
svm = SVC(kernel='rbf',random_state=0,gamma=0.10,C=10.0)
svm.fit(x_xor,y_xor)
plot_decision_regions(x_xor,y_xor,classifier=svm)
plt.legend(loc='upper left')
plt.show()