

form sklearn.cross_validation import train_test_split
x = df.iloc[:,:-1].values
y = df.['MEDV'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
slr = LinearRegression()
slr.fit(x_train,y_train)
y_train_pred = slr.predict(x_train)
y_test_pred = slr.predict(x_test)

# 我们无法在二维图上绘制线性回归曲线（更确切地说是超平面），不过可以绘制出预测值的残差（真实值与预测值之间的差异或者垂直距离）图，从而对回归模型进行评估。
plt.scatter(y_train_pred,y_train_pred-y_train,c='blue',marker='0',label='Training data')
plt.scatter(y_test_pred,y_test_pred-y_test,c='lightgreen',marker='s',balel='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.show()
# 完美的预测结果其残差应为0，但在实际应用中，这种情况可能永远都不会发生。


# 另外一种对模型性能进行定量评估的方法称为均方误差（Mean Squared Error，MSE），它是线性回归模型拟合过程中，最小化误差平方和（SSE）代价函数的平均值。MSE可用于不同回归模型的比较，或是通过网格搜索进行参数调优，以及交叉验证等：
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train,y_train_pred),mean_squared_error(y_test,y_test_pred)))


# R2
from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train,y_train_pred),
r2_score(y_test,y_test_pred)))
