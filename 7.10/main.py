import pandas as pd
import math


def EOperation(score, seta1, seta2, seta3):
        total = seta1*(1-seta1) + seta2*(1-seta2) + seta3*(1-seta3)
        aerfa = seta1*(1-seta1)/total
        segema = seta2*(1-seta2)/total
        gama = seta3*(1-seta3)/total
        seta1_H = aerfa * score
        seta1_T = (1-aerfa) * score
        seta2_H = segema * score
        seta2_T = (1-segema) * score
        seta3_H = gama * score
        seta3_T = (1-gama) * score



def MOperation(dataset, seta1, seta2, seta3):
        pass


dataset = pd.read_csv('data/DataSet.csv')

#do not care about the score. Development first.
score = {'dark_green': 1, 'black': 2, 'light_white': 3,
         'curl_up': 1, 'little_curl_up': 2, 'stiff': 3,
         'little_heavily': 1, 'heavily': 2, 'clear': 3,
         'distinct': 1, 'little_blur': 2, 'blur': 3,
         'sinking': 1, 'little_sinking': 2, 'even': 3,
         '1': 1, '0': 0}

# #使用one-hot编码--不行，因为转换后所有的属性有值的都是1
# wm_df = pd.get_dummies(dataset)
# print(wm_df)

del dataset['Idx']  # 删除某一列，但是不推荐这么做。可以使用drop。
del dataset['density']
del dataset['sugar_ratio']
del dataset['touch']
colcount = dataset.columns.size
for i in range(colcount-1):
        #实用映射的方式来转换
        dataset[dataset.columns.values[i]
                ] = dataset[dataset.columns.values[i]].map(score)
print(dataset)

last_seta1 = 0.0
last_seta2 = 0.0
last_seta3 = 0.0
seta1 = 0.0
seta2 = 0.0
seta3 = 0.0
frist = True
while fabs(last_seta1-seta1) < 0.1 and fabs(last_seta2-seta2) < 0.1 and fabs(last_seta3-seta3) < 0.1:
        if not frist:
                seta1, seta2, seta3 = EOperation(dataset, seta1, seta2, seta3)
        MOperation(dataset, seta1, seta2, seta3)
        frist = False
print('account over.')
print(seta1, seta2, seta3)
