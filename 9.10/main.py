#试着编程出不需要人工输入的初始点集

import pandas as pd
import numpy as np  # 主要操作对象使矩阵
import random
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist  

class myobject:
    '''
    一个中心点+N个周围点构成的集合
    '''

    def __init__(self, density, sugarrate):
        self.__density = density
        self.__sugarrate = sugarrate
        self.__values = [(density, sugarrate)]
        self.__originaldensity = density
        self.__originalsugarrate = sugarrate

    def push(self, density, sugarrate):
        self.__values.append((density, sugarrate))
        m = np.array(self.__values)
        self.__density = np.mean(m[:, 0])
        self.__sugarrate = np.mean(m[:, 1])

    def dis(self, x, y):
        return math.sqrt(math.pow(x-self.__density, 2) + math.pow(y - self.__sugarrate, 2))

    def __str__(self):  # 定义打印对象时打印的字符串---当对象直接调用的时候调用的使这个
        return " ".join(str(item) for item in (
            self.__density, self.__sugarrate))

    def __repr__(self):  # 当实用容器来调用的时候，调用的使这个
        return " ".join(str(item) for item in (
            self.__density, self.__sugarrate))

    def getoriginalcenter(self):
        return self.__originaldensity, self.__originalsugarrate

    def getcenter(self):
        m = np.array(self.__values)
        return np.mean(m[:, 0]),np.mean(m[:, 1])

    def getcol(self, i):
        m = np.array(self.__values)
        return m[:, i]

    def accumulate(self):
        m = np.array(self.__values)
        X = pdist(m, 'euclidean')
        return np.mean(X)


def kmeans(dataset, centers):
    for i in range(len(dataset)):
        # this is the right way to foreach each item of list.
        dislist = [x.dis(dataset.ix[i, [1]], dataset.ix[i, [2]])
                   for x in centers]
        matchcenter = dislist.index(min(dislist))
        centers[matchcenter].push(dataset.ix[i, [1]], dataset.ix[i, [2]])
    avgdis = []
    avgcenter = []
    score = []
    for i in range(len(centers)-1):
        x,y = centers[i].getcenter()
        x1,y1 = centers[i+1].getcenter()
        avgcenter.append([x,y])
        avgcenter.append([x1,y1])
        dis = pdist(avgcenter,'euclidean')
        avgdis.append(centers[i].accumulate())
        score.append()


dataset = pd.read_csv('data/melondataset9.csv').iloc[:, 0:3]
forcondition = True
while(forcondition):
    kmeans_K = 2
    centers = []
    for i in range(0, kmeans_K):
        centers.append(myobject(random.uniform(0, 1), random.uniform(0, 1)))
    kmeans(dataset, centers)
    