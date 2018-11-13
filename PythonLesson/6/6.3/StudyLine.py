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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

pipe_lr = Pipeline([('scl',StandardScaler()),
    ('clf',LogisticRegression(penalty='l2',random_state=0))])

# 通过learning_curve函数的train_size参数，我们可以控制用于生成学习曲线的样本的绝对或相对数量。
# 通过设置train_sizes=np.linspace（0.1,1.0,10）来使用训练数据集上等距间隔的10个样本。
train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=x_train,y=y_train,train_sizes=np.linspace(0.1,1.0,10),n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.5,color='blue')
plt.plot(train_sizes,test_mean,color='green',marker='^',markersize=5,label='Validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.5,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.0])
plt.show()