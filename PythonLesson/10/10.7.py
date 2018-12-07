import numpy as np

# 线性回归模型的曲线化-多项式回归
from sklearn.preprocessing import PolynomialFeatures
# 增加一个二次多项式项：
x = np.array([258.0,270.0,294.0,
                320.0,342.0,368.0,
                396.0,446.0,480.0,
                586.0])[:,np.newaxis]
y=np.array([236.4,234.4,252.8,
            298.6,314.2,342.2,
            360.8,368.0,391.2,
            390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degress=2)
x_quad = quadratic.fit_transform(x)

# 拟合一个用于对比的简单线性回归模型：
lr.fit(x,y)
x_fit = np.arange(250,600,10)[:,np.newaxis]
y_lin_fit = lr.predict(x_fit)

# 使用经过转换后的特征针对多项式回归拟合一个多元线性回归模型：
pr.fit(x_quad,y)
y_quad_fit=pr.predict(quadratic.fit_transform(x_fit))

# plot the results:
plt.scatter(x,y,label='training points')
plt.plot(x_fit,y_lin_fit,label='linear fir',linestyle='--')
plt.plot(x_fit,y_quad_fit,label='quadratic fit')
plt.legend(loc='upper left')
plt.show()

y_lin_pred = lr.predict(x)
y_quad_pred = pr.predict(x_quad)
print('Training MSE linear: %.3f, quadratic: %.3f' % (mean_squared_erro(y,y_lin_pred),
mean_squared_error(y,y_quad_pred)))
print('Training R^2 linear: %.3f, quadratic: %.3f' % (r2_score(y,y_lin_pred),r2_score(y,y_quad_pred)))