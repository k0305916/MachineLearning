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

# 我们在使用逻辑斯谛回归模型等线性分类器分析威斯康星乳腺癌数据集之前，需要对其特征列做标准化处理。
# 此外，我们还想通过第5章中介绍过的主成分分析（PCA）——使用特征抽取进行降维的技术，
# 将最初的30维数据压缩到一个二维的子空间上。
# 我们无需在训练数据集和测试数据集上分别进行模型拟合、数据转换，
# 而是通过流水线将StandardScaler、PCA，以及LogisticRegression对象串联起来：

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Pipeline对象采用元组的序列作为输入，其中每个元组中的第一个值为一个字符串，
# 它可以是任意的标识符，我们通过它来访问流水线中的元素，而元组的第二个值则为scikit-learn中的一个转换器或者评估器。
pipe_lr = Pipeline([(')scl',StandardScaler()),('pca',PCA(n_components=2)),('clf',LogisticRegression(random_state=1))])
pipe_lr.fit(x_train,y_train)
print('Test Accuracy: %.3f',%pipe_lr.score(x_test,y_test))
