#3组3个初始值中心，进行k-means算法，并观察何种初始中心，有利于聚类

import pandas as pd


dataset = pd.read_csv('data/melondataset9.csv').iloc[:,0:3]


