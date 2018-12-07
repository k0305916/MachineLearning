


import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',header=None,sep='\s+')
df.columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PREATIO','B','LSTAT','MEDV']
# print(df.head())

# 借助散点图矩阵，我们以可视化的方法汇总显示各不同特征两两之间的关系。
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style='whitegrid',context='notebook')  #与下面的显示图冲突，需要注释掉
cols=['LSTAT','INDUS','NOX','RM','MEDV']
# sns.pairplot(df[cols],size=2.5)
# sns.reset_orig()//恢复matplotlib的风格
# plt.show()

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,
                fmt='.2f',annot_kws={'size':15},yticklabels=cols,xticklabels=cols)
# plt.show()


class LinearRegressionGD(object):
    def __init__(self,eta=0.001,n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self,x,y):
        self.w_ = np.zeros(1+x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = (y-output)
            self.w_[1:]+=self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self,x):
        return np.dot(x,self.w_[1:])+self.w_[0]

    def predict(self,x):
        return self.net_input(x)

x = df[['RM']].values
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_std = sc_x.fit_transform(x)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(x_std,y_std)

# plt.plot(range(1,lr.n_iter+1),lr.cost_)
# plt.ylabel('SSE')
# plt.xlabel('Epoch')
# plt.show()

def lin_regplot(x,y,model):
    plt.scatter(x,y,c='blue')
    plt.plot(x,model.predict(x),color='red')
    return None

lin_regplot(x_std,y_std,lr)
# plt.xlable('Average number of rooms[RM](standardized')
# plt.ylable('Price in $1000\'s [MEDV] (standardized)')
# plt.show()

num_rooms_std = sc_x.transform([5.0])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's %.3f" % sc_y.inverse_transform(price_std))


# 不过在实际应用中，我们可能会更关注如何高效地实现模型，例如，scikit-learn中的LinearRegression对象使用了LIBLINEAR库以及先进的优化算法，可以更好地使用经过标准化处理的变量。这正是特定应用所需要的：
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(x,y)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

lin_regplot(x,y,slr)
# plt.xlable('Average number of rooms[RM](standardized')
# plt.ylable('Price in $1000\'s [MEDV] (standardized)')
# plt.show()

