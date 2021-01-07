from scipy.signal import lfilter
from spectrum import *
from numpy.random import randn
import matplotlib.pyplot as plt



A  =[1, -2.7607, 3.8106, -2.6535, 0.9238]
noise = randn(1, 1024)
y = lfilter([1], A, noise)
#filter a white noise input to create AR(4) process
count = 20
[ar, var, reflec] = aryule(y[0], count)
# ar should contains values similar to A


# yhat = np.concatenate(y[0], ar)
# plt.plot(y[0])
# plt.plot(yhat)
plt.plot(noise[0])
plt.plot(y[0])

# id = np.arange(0, count, 1)
# plt.plot(ar)
# plt.scatter(id, ar)

plt.show()


print("over")