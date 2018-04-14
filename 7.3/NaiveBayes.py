
import pandas as pd
import numpy as np

def InportData():
    pass


def Lapulasi():
    pass


def CalcClassProbability():
    pass

def CalcClassConditionProbability():
    pass

def CalcProbability(df):
    Probability = {'是':{},'否':{}}
    # 计算p(c)
    dfclass = df.groupby(df.columns[-1]).count()

    # 计算对应的元素的概率p(x|c).
    for col in np.arange(1,df.columns.size-1,1):
        dftemp = df.iloc[0:,[col,-1]]
        dftemp['count'] = np.ones(dftemp.iloc[:,0].size)
        #这里有个更好的表示就更好了。[dftemp.columns[0],dftemp.columns[1]]
        dftemp = dftemp.groupby([dftemp.columns[0],dftemp.columns[1]]).count()
        test = dftemp[0:1]
        #print(dftemp)
        # dftemp['Probability'] = dftemp.apply(lambda value: print(value.index))
        # dftemp.describe()
        print(test.index.levels)
    



def main():
    # data_file_encode = "gb18030"
    with open("data//watermelon_3.csv", mode = 'r') as data_file:
        df = pd.read_csv(data_file)
    CalcProbability(df)
    





if __name__ == "__main__":
    main()