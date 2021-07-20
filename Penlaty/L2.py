import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
from sklearn.datasets import load_boston
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge



boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
#print boston_df.info()
# add another column that contains the house prices which in scikit learn datasets are considered as target
boston_df['Price']=boston.target
#print boston_df.head(3)
newX=boston_df.drop('Price',axis=1)
newX=newX.drop('CRIM',axis=1)
newX=newX.drop('ZN',axis=1)
newX=newX.drop('INDUS',axis=1)
newX=newX.drop('CHAS',axis=1)
newX=newX.drop('NOX',axis=1)
newX=newX.drop('RM',axis=1)
newX=newX.drop('AGE',axis=1)
newX=newX.drop('DIS',axis=1)
newX=newX.drop('RAD',axis=1)
newX=newX.drop('B',axis=1)
newX=newX.drop('LSTAT',axis=1)

# newX=boston_df.iloc[:,:2]
# newX.reshape(1, -1)
print(newX[0:3]) # check 
newY=boston_df['Price']
#print type(newY)# pandas core frame
X_train,X_test,y_train,y_test=train_test_split(newX,newY,test_size=0.3,random_state=3)
print(len(X_test), len(y_test))
lr = LinearRegression()
lr.fit(X_train, y_train)
rr = Ridge(alpha=0.01) 
# higher the alpha value, more restriction on the coefficients; low alpha > more generalization,
# in this case linear and ridge regression resembles
rr.fit(X_train, y_train)
rr100 = Ridge(alpha=100) #  comparison with alpha value
rr100.fit(X_train, y_train)
print(lr.coef_)
print(rr.coef_)
print(rr100.coef_)
# train_score=lr.score(X_train, y_train)
# test_score=lr.score(X_test, y_test)
# Ridge_train_score = rr.score(X_train,y_train)
# Ridge_test_score = rr.score(X_test, y_test)
# Ridge_train_score100 = rr100.score(X_train,y_train)
# Ridge_test_score100 = rr100.score(X_test, y_test)
# plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Ridge; $\alpha = 0.01$',zorder=7) 
# plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge; $\alpha = 100$') 
# plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')
# plt.xlabel('Coefficient Index',fontsize=16)
# plt.ylabel('Coefficient Magnitude',fontsize=16)
# plt.legend(fontsize=13,loc=4)
# plt.show()