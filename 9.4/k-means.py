#3组3个初始值中心，进行k-means算法，并观察何种初始中心，有利于聚类

import pandas as pd
import numpy as np  # 主要操作对象使矩阵
import random
import math
import matplotlib.pyplot as plt


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


def kmeans(dataset, centers):
    for i in range(len(dataset)):
        # this is the right way to foreach each item of list.
        dislist = [x.dis(dataset.ix[i, [1]], dataset.ix[i, [2]])
                   for x in centers]
        matchcenter = dislist.index(min(dislist))
        centers[matchcenter].push(dataset.ix[i, [1]], dataset.ix[i, [2]])


dataset = pd.read_csv('data/melondataset9.csv').iloc[:, 0:3]
for i in range(0, 3):
    centers = []
    for i in range(0, 3):
        centers.append(myobject(random.uniform(0, 1), random.uniform(0, 1)))
    kmeans(dataset, centers)

    # draw scatter diagram to show the raw data
    plt.figure(i)
    plt.title('k-means')
    plt.xlabel('density')
    plt.ylabel('sugar_rate')
    plt.scatter(centers[0].getcol(0), centers[0].getcol(1),
                marker='o', color='k', s=5, label='1')
    x, y = centers[0].getcenter()
    plt.plot(x, y, 'k+')
    x, y = centers[0].getoriginalcenter()
    plt.plot(x, y, 'k*')
    plt.scatter(centers[1].getcol(0), centers[1].getcol(1),
                marker='o', color='g', s=5, label='2')
    x, y = centers[1].getcenter()
    plt.plot(x, y, 'g+')
    x, y = centers[1].getoriginalcenter()
    plt.plot(x, y, 'g*')
    plt.scatter(centers[2].getcol(0), centers[2].getcol(1),
                marker='o', color='r', s=5, label='3')
    x, y = centers[2].getcenter()
    plt.plot(x, y, 'r+')
    x, y = centers[2].getoriginalcenter()
    plt.plot(x, y, 'r*')
    # plt.legend(loc='upper right')
    plt.show()
    # print(centers)
