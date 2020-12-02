# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import scipy as scipy
from scipy import optimize
# from matplotlib.ticker import AutoMinorLocator
# from matplotlib import gridspec
# import matplotlib.ticker as ticker

# x_array = np.linspace(1,10,10)
# y_array_2exp = (np.exp(-x_array*0.1) + np.exp(-x_array*1))

# define a fitting function called exponentail which takes
# in the x-data (x) and returns an exponential curve with equation
# a*exp(x*k) which best fits the data
def exponential(x, a, b, c):
    # return a*np.exp(x*k1) + b*np.exp(x*k2) + c
    return a * np.exp(b * x[0] + c * x[1])

def exponential1(x, a, b, c):
    # return a*np.exp(x*k1) + b*np.exp(x*k2) + c
    return a + b * x[0] + c * x[1]

# def _2exponential(x, a, k1, b, k2, c):
    # return a*np.exp(x*k1) + b*np.exp(x*k2) + c

def _exponential(x, a, b, c):
    x0 = np.asarray(x[0])
    x1 = np.asarray(x[1])
    value = b * x0
    value = value + c* x1
    return a * np.exp(value)

def _exponential1(x, a, b, c):
    x = np.asarray(x)
    return a + b * x[0] + c * x[1]

y = [0.5,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
x = [[0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.19,0.11,0.12],[0.33,0.34,0.36,0.35,0.37,0.38,0.39,0.31,0.32,0.33]]

# popt_2exponential, pcov_2exponential = scipy.optimize.curve_fit(_2exponential, x_array, y_array_2exp, p0=[1,-0.1, 1, -1, 1])

# using the scipy library to fit the x- and y-axis data 
# p0 is where you give the function guesses for the fitting parameters
# this function returns:
#   popt_exponential: this contains the fitting parameters
#   pcov_exponential: estimated covariance of the fitting paramters
# original exponention regression
popt, pcov = scipy.optimize.curve_fit(exponential, x, y)
fitted = _exponential(x, popt[0], popt[1], popt[2])
print(("a: {0}, b: {1}, c: {2}").format(popt[0],popt[1],popt[2]))
print(fitted)

# PAL exponential regression
# original exponention regression
popt, pcov = scipy.optimize.curve_fit(exponential1, x, y)
# popt[0] = np.exp(popt[0])
lnfitted = _exponential1(x, popt[0], popt[1], popt[2])
print(lnfitted)

fitted = np.exp(lnfitted)
print(("a: {0}, b: {1}, c: {2}").format(popt[0],popt[1],popt[2]))
print(fitted)
