# 我们使用scikit-learn中的PCA对葡萄酒数据集做预处理，
# 然后使用逻辑斯谛回归对转换后的数据进行分类，
# 最后用第2章中定义的plot_decision_region函数对决策区域进行可视化展示。

from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def plot_decision_regions(x,y,classifier,resolution=0.02):
    #setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors=('red','blue','lightgreen','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])

    #plot the decision surface
    x1_min,x1_max = x[:,0].min()-1,x[:,0].max()+1
    x2_min,x2_max = x[:,1].min()-1,x[:1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
        np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha = 0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    #plot class samples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=cl)


df_wine = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)

# standardScaler---the first PCA step.
stdsc = StandardScaler()
x_train_std = stdsc.fit_transform(x_train)
x_test_std = stdsc.transform(x_test)

    
pca = PCA(n_components=2)
lr = LogisticRegression()
x_train_pca = pca.fit_transform(x_train_std)
x_test_pca = pca.transform(x_test_std)
lr.fit(x_train_pca,y_train)
plot_decision_regions(x_train_pca,y_train,classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

# 为了保证整个分析过程的完整性，
# 我们绘制一下逻辑斯谛回归在转换后的测试数据上所得到的决策区域，看其是否能很好地将各类分开：

plot_decision_regions(x_test_pca,y_test,classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

# 如果我们对不同主成分的方差贡献率感兴趣，可以将PCA类中的n_componets参数设置为None，
# 由此，可以保留所有的主成分，并且可以通过explained_variance_ratio_属性得到相应的方差贡献率：
# 请注意：在初始化PCA类时，如果我们将n_components设置为None，
# 那么它将按照方差贡献率递减顺序返回所有的主成分，而不是进行降维操作。

pca = PCA(n_components=None)
x_train_pca = pca.fit_transform(x_train_std)
print(pca.explained_variance_ratio_)
