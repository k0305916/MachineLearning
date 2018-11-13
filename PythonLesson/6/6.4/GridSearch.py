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
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

pipe_svc = Pipeline([('scl',StandardScaler()),
                ('clf',SVC(random_state=1))])
param_range=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid=[{'clf__C':param_range,
            'clf__kernel':['linear']},
            {'clf__C':param_range,
            'clf__gamma':param_range,
            'clf__kernel':['rbf']}]
gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=-1)
gs = gs.fit(x_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

# 虽然网格搜索是寻找最优参数集合的一种功能强大的方法，
# 但评估所有参数组合的计算成本也是相当昂贵的。
# 使用scikit-learn抽取不同参数组合的另一种方法就是随机搜索（randomized search）。
# 借助于scikit-learn中的RandomizedSearchCV类，我们可以以特定的代价从抽样分布中抽取出随机的参数组合。
# 关于此方法的更详细细节及其示例请访问链接：
# http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization。
clf = gs.best_estimator_
clf.fit(x_train,y_train)
print('Test accuary: %.3f' % clf.score(x_test,y_test))