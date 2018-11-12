import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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

# 我们将使用NumPy中的linalg.eig函数来计算葡萄酒数据集协方差矩阵的特征对：--the second PCA step.
cov_mat = np.cov(x_train_std.T)
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' %eigen_vals)

# 非PCA步骤，但是为了方便观看
# 使用NumPy的cumsum函数，我们可以计算出累计方差，其图像可通过matplotlib的step函数绘制：
# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals,reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1,14),var_exp,alpha=0.5,align='center',label='individual explained variance')
# plt.step(range(1,14),cum_var_exp,where='mid',label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal components')
# plt.legend(loc='best')
# plt.show()

# Descending oder---the third PCA Step
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

# 选择新的特性
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
                eigen_pairs[1][1][:,np.newaxis]))
print('Matrix W:\n',w)

# 映射到新的特征空间中
# x_train_std[0].dot(w)
x_train_pca = x_train_std.dot(w)

# visualization
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(x_train_pca[y_train==l,0],
    x_train_pca[y_train==l,1],
    c=c,label=l,marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

