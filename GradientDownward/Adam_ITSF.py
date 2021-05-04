from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed
import numpy as np
import pandas as pd
 
g_p_gradient = 0.0
g_q_gradient = 0.0
# objective function
def ITSF(
        input_series, 
        input_series_length,
        w, 
        h,                  
        epsilon,
        gradient
        ):
    zfit = np.array([None] * input_series_length)
    xfit = np.array([0.0] * input_series_length)
    
    if len(w) < 2:
        raise Exception('This is the exception you expect to handle')
    else:
        p = w[0]
        q = w[1]

        p_gradient = gradient[0]
        q_gradient = gradient[1]

    # fit model
    cc = []
    nt = []
    j = 1
    k = 1
    p_gradientlist = []
    q_gradientlist = []
    for i in range(0,input_series_length):
        nt.append(k)
        if input_series[i] == 0:
            k += 1
            if i == 0:
                zfit[i] = 0.0
                xfit[i] = 0.0
            else:
                zfit[i] = zfit[i-1]
                xfit[i] = xfit[i-1]
        else:
            a_demand = p / j
            a_interval = q / j

            if i <= p:
                zfit[i] = input_series[i]
            else:
                zfit[i] = a_demand * input_series[i] + (1 - a_demand) * zfit[i-1]

            if i == 0:
                xfit[i] = 0.0
            elif i <= q:
                xfit[i] = k
            else:
                xfit[i] = a_interval * k + (1-a_interval) * xfit[i-1]

            k = 1

        if xfit[i] == 0:
            cc.append(zfit[i])
        else:
            cc.append(zfit[i] / (xfit[i]+1e-7))
        j+=1

        if i > 0 :
            aa = 0
            bb = 0
            if ((k - xfit[i-1]) * q + j * xfit[i-1]) != 0.0 and input_series[i] != 0 :
                p_gradientlist.append((input_series[i] - zfit[i-1]) / ((k - xfit[i-1]) * q + j * xfit[i-1]))
                aa = (input_series[i] - zfit[i-1]) / ((k - xfit[i-1]) * q + j * xfit[i-1])
            if ((k - xfit[i-1]) * q + j * xfit[i-1]) != 0.0 and input_series[i] != 0 :
                q_gradientlist.append(((input_series[i] - zfit[i-1]) * p + j * zfit[i-1]) * (xfit[i-1] - k) / (((k - xfit[i-1]) * q + j * xfit[i-1]) ** 2))
                bb = ((input_series[i] - zfit[i-1]) * p + j * zfit[i-1]) * (xfit[i-1] - k) / (((k - xfit[i-1]) * q + j * xfit[i-1]) ** 2)

            # print(("----p_gradient: {0}   q_gradient: {1}").format(aa, bb))

    p_gradientlist.pop(0)
    q_gradientlist.pop(0)
    p_gradient = sum(p_gradientlist)
    q_gradient = sum(q_gradientlist)
    # print(("p: {0} q: {1} last interval: {2}").format(p, q, xfit[-1]))
           
    ata_model = {
                        'a_demand':             p,
                        'a_interval':           q,
                        'demand_series':        pd.Series(zfit),
                        'interval_series':      pd.Series(xfit),
                        'demand_process':       pd.Series(cc),
                        'in_sample_nt':         pd.Series(nt),
                        'last_interval':        xfit[-1],
                        'gredient':             (p_gradient, q_gradient)
                    }
    # print(zfit)
    # print(xfit)
    # calculate in-sample demand rate
    frc_in = cc

    # forecast out_of_sample demand rate
    if h > 0:
        frc_out = []
        a_demand = p / input_series_length
        a_interval = q / input_series_length
        zfit_frcout = a_demand * input_series[-1] + (1-a_demand)*zfit[-1]
        xfit_frcout = a_interval * nt[-1] + (1-a_interval)*xfit[-1]


        for i in range(1,h+1):
            result = zfit_frcout / xfit_frcout
            frc_out.append(result)

        # print(('frc_out: {0}').format(frc_out))
        # f.write(str(zfit_frcout) +'\t' + str(xfit_frcout) + '\t' + str(result) + '\n')
    else:
        frc_out = None

    # f.close()
    return_dictionary = {
                            'model':                    ata_model,
                            'in_sample_forecast':       frc_in,
                            'out_of_sample_forecast':   frc_out,
                            'fit_output':               cc,
                            'gredient':                 (p_gradient, q_gradient)
                        }
    
    return return_dictionary
 
def objective(
            p0,
            input_series,
            input_series_length,
            epsilon
            ):

    # p = p0[0]
    # q = p0[1]
    # if q > p:
    #     return 99999999999999.999

    global g_p_gradient
    global g_q_gradient
    # -------------------------------------------------------------
    ata_result = ITSF(
                    input_series = input_series,
                    input_series_length = input_series_length,
                    w=p0,
                    h=0,
                    epsilon = epsilon,
                    gradient= (0, 0)
                    )
    frc_in = ata_result['in_sample_forecast']
    # print(frc_in)
    (g_p_gradient, g_q_gradient) = ata_result['gredient']

    E = input_series - frc_in

    # standard RMSE
    E = E[E != np.array(None)]
    # E = np.sqrt(np.mean(E ** 2))
    E = np.mean(E ** 2)

    print(("p_gradient: {0}   q_gradient: {1}").format(g_p_gradient, g_q_gradient))
    if len(p0) < 2:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], 0.0, E))
    else:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[1], E))
    return E

# derivative of objective function
def derivative(x, y):
    return (g_p_gradient, g_q_gradient)
	
 
# gradient descent algorithm with adam
def adam(input_series, input_series_length, objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
	# generate an initial point
	# x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    x = np.asarray((1.0, 1.0))
    score = objective(x, input_series, input_series_length, 1e-7)
    # initialize first and second moments
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]
    # run the gradient descent updates
    for t in range(n_iter):
        # calculate gradient g(t)
        g = derivative(x[0], x[1])
        # build a solution one variable at a time
        for i in range(x.shape[0]):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
            # mhat(t) = m(t) / (1 - beta1(t))
            mhat = m[i] / (1.0 - beta1**(t+1))
            # vhat(t) = v(t) / (1 - beta2(t))
            vhat = v[i] / (1.0 - beta2**(t+1))
            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
        # evaluate candidate point
        score = objective(x, input_series, input_series_length, 1e-7)
        # report progress
        print('>%d f(%s) = %.5f' % (t, x, score))
    return [x, score]


ts = [362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
      0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
      0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
      0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
      0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
      0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
      0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
      312.74, 0.0, 0.0, 0.0, 0.0]
 
# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[1.0, 82.0], [1.0, 82.0]])
# define the total iterations
n_iter = 60
# steps size
alpha = 0.02
# factor for average gradient
beta1 = 0.8
# factor for average squared gradient
beta2 = 0.999
# perform the gradient descent search with adam
best, score = adam(np.asarray(ts), len(ts), objective, derivative, bounds, n_iter, alpha, beta1, beta2)
print('Done!')
print('f(%s) = %f' % (best, score))
