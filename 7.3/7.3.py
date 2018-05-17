import pandas as pd


def AccountPd(dataset):
    '''
    取出最后一列，得到好瓜/坏瓜的总数以及比例
    '''
    pgood = len([x for x in dataset if x == 1]) / len(dataset)
    pbad = len([x for x in dataset if x == 0]) / len(dataset)
    return pgood, pbad


def AccountPx(dataset,value):
    '''
    计算（属性值(i,j...)，好瓜）的值，以及（属性值(i,j...)，坏瓜）的值
    '''
    pgood = dataset[dataset['label']==1].size/dataset.size
    pbad = dataset[dataset['label']==0].size/dataset.size
    return pgood, pbad


dataset = pd.read_csv('data/DataSet.csv')
testdataset = ['dark_green','curl_up','little_heavily',]
del dataset['Idx'] #删除某一列，但是不推荐这么做。可以使用drop。
colcount = dataset.columns.size
resultgood = 1.0
resultbad = 1.0
for i in range(colcount):
    newdataset = pd.DataFrame()
    newdataset.insert(0, 'property', dataset[dataset.columns[i]])
    newdataset.insert(1, 'label', dataset[dataset.columns[-1]])
    pgood, pbad = AccountPx(newdataset)
    resultgood *= pgood
    resultbad *= pbad

#通过这种方式得到的结果是个列表[]
pgood, pbad = AccountPd(dataset[dataset.columns[-1]])
result1 = resultgood*pgood
result2 = resultbad*pbad
if result1 > result2:
    print('good')
else:
    print('bad')
