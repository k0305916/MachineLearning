import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

def f(x):   # The rosenbrock function
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2
def jacobian(x):
    # return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))
    return np.array((-2*.5*(1 - x[0]), 2*(x[1] - x[0]**2)))

result = scipy.optimize.minimize(f, [2, 2], method="L-BFGS-B", jac=None)
print(result)

result = scipy.optimize.brute(f,((-1, 2), (-1, 2)))
print(result)