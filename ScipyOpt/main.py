import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
# from operator import itemgetter

def f(x):   # The rosenbrock function
    print(("x0: {0}, x1: {1}").format(x[0],x[1]))
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
def jacobian(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    # return np.array((-2*.5*(1 - x[0]), 2*(x[1] - x[0]**2)))

result = scipy.optimize.minimize(f, [2, 2], method='L-BFGS-B', jac=None)
# result = scipy.optimize.minimize(f, [2, 2], method='L-BFGS-B', jac=jacobian)
print(result)

# result = scipy.optimize.brute(f,((-1, 2), (-1, 2)))
# print(result)

# opt_para = [{'para': (0,1), 'MSE':0.22},
#             {'para': (2,1), 'MSE':0.44},
#             {'para': (1,0), 'MSE':0.11}]

# # opt_para.sort(key=itemgetter('MSE'))
# opt_para = sorted(opt_para, key=lambda item: item['MSE'])
# print(opt_para)