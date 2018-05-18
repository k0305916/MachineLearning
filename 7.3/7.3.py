import pandas as pd
import math


def AccountPd(dataset):
    '''
    取出最后一列，得到好瓜/坏瓜的总数以及比例
    '''
    N = len(set(dataset))
    pgood = (len([x for x in dataset if x == 1])+1) / (len(dataset)+N)
    pbad = (len([x for x in dataset if x == 0])+1) / (len(dataset)+N)
    return pgood, pbad


def AccountPx(dataset, value):
    '''
    计算（属性值(i,j...)，好瓜）的值，以及（属性值(i,j...)，坏瓜）的值
    '''
    # if is str class
    if isinstance(value, str):
        #下面是错误的，需要修改成属性的可能个数，而不是属性的数据个数
        Ni = dataset['property'].count
        pgood = (dataset[(dataset['label'] == 1) & (
            dataset['property'] == value)].size + 1)/ (dataset[dataset['label'] == 1] + Ni)
        pbad = (dataset[dataset['label'] == 0 & (
            dataset['property'] == value)].size + 1)/ (dataset[dataset['label'] == 0] + Ni)
    else:  # if is double class
        miugood = dataset[['property']][dataset['label']== 1].sum() / dataset.size
        sigmagood = dataset[['property']][dataset['label'] == 1].apply(
            lambda x: (x - miugood)*(x - miugood)).sum() / dataset[dataset['label'] == 1].size
        miubad = dataset[['property']][dataset['label']== 0].sum() / dataset.size
        sigmabad = dataset[['property']][dataset['label'] == 0].apply(
            lambda x: (x - miugood)*(x - miugood)).sum() / dataset[dataset['label'] == 0].size
        pgood = 1 / (math.sqrt(2*math.pi) * miugood) * math.exp(-(value -
                sigmagood) * (value - sigmagood) / (2*miugood*miugood))
        pbad = 1 / (math.sqrt(2*math.pi) * miubad) * math.exp(-(value -
                sigmabad) * (value - sigmabad) / (2*miubad*miubad))
    return pgood, pbad


dataset = pd.read_csv('data/DataSet.csv')
#dark_green,curl_up,little_heavily,distinct,sinking,hard_smooth,0.697,0.46
testdataset = ['dark_green', 'curl_up', 'little_heavily',
               'distinct', 'sinking', 'hard_smooth', 0.697, 0.46]
del dataset['Idx']  # 删除某一列，但是不推荐这么做。可以使用drop。
colcount = dataset.columns.size
resultgood = 1.0
resultbad = 1.0
for i in range(colcount-1):
    newdataset = pd.DataFrame()
    newdataset.insert(0, 'property', dataset[dataset.columns[i]])
    newdataset.insert(1, 'label', dataset[dataset.columns[-1]])
    pgood, pbad = AccountPx(newdataset, testdataset[i])
    resultgood *= pgood
    resultbad *= pbad

'''
整体的流程写出来了，但是计算的结果不对。。。
先不管这个了。。。继续下面的作业。
'''
#通过这种方式得到的结果是个列表[]
pgood, pbad = AccountPd(dataset[dataset.columns[-1]])
result1 = resultgood*pgood
result2 = resultbad*pbad
print(result1)
print(result2)
if result1 > result2:
    print('good')
else:
    print('bad')
