from math import fabs
import numpy as np
import random

def intervalFix(data, interval, count, demandfix = False):
    dataset = []
    endrange = interval * count
    _data = data
    for i in range(0, endrange, interval):
        if demandfix == True:
            _data = data - random.randint(0, data - 1)
        dataset.append(_data)
        dataset =  dataset + [0] * (interval - 1 )
    return dataset

def intervalsmooth(data, interval, count, step = 0, demandfix = False):
    datasetlist = []
    for i in range(count):
        datasetlist.append([0] * (i+interval+step))

    _data = data
    for i in range(len(datasetlist)):
        if demandfix == True:
            _data = data - random.randint(0, data - 1)
        datasetlist[i].insert(0, _data)

    return sum(datasetlist, [])


def intervalrandom(data, interval, count, demandfix = False):
    dataset = []
    _data = data
    for i in range(count):
        if demandfix == True:
            # _data = data - random.randint(0, data - 1)
            _data = random.randint(data - 100, data - 1)
            # print(_data)
        _interval = random.randint(0, interval)

        dataset.append(_data)
        dataset = dataset + [0] * _interval

    return dataset

def intervaldecrease(data, interval, count, step = 0, demandfix = False):
    datasetlist = []
    for i in range(count):
        datasetlist.append([0] * abs((abs(interval-i) - step)))

    _data = data
    for i in range(len(datasetlist)):
        if demandfix == True:
            _data = data - random.randint(0, data - 1)
        datasetlist[i].insert(0, _data)

    return sum(datasetlist, [])


# _dataset = intervalFix(1000, 4, 810, True)
# _dataset = intervalsmooth(1000, 1, 30, 0, True)
_dataset = intervalrandom(1000, 3, 30, True)
# _dataset = intervaldecrease(1000, 30, 10, 1, True)
print(_dataset)


# observation = [923, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 900, 921, 0, 0, 0, 0, 0, 0, 0, 983, 0, 0, 963, 0, 0, 0, 0, 0, 989, 0, 0, 0, 0, 0, 951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 964, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 944, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 985, 0, 0, 0, 0]
# print(len(observation))
# result1 = [0, 0, 0, 0, 0, 0, 321, 0, 0, 0, 0, 0, 0, 0, 328.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 326.605, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 337.923, 0, 337.625, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 338.421, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 338.91, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 339.375, 0, 0, 344.859, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 345.165, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 344.544, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 345.92, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 345.721, 0, 0, 347.159, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 346.075, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 347.577, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 348.852, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 350.372, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 351.171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 351.12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 351.097, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 351.977, 0, 0, 0, 0, 0, 0, 352.244, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 353.753, 353.758, 0, 0, 0, 0, 0, 0, 0, 354.659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 355.142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 354.977, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 354.798, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


# mse1 = np.sum((np.array(observation) - np.array(result1)) * (np.array(observation) - np.array(result1))) / len(observation)
# print("{0} {1} mse1: {2}".format(len(observation), len(result1), mse1))
