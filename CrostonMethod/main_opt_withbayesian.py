import numpy as np
import pandas as pd
# from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

def fit_croston(
                    input_endog,
                    forecast_length,
                    croston_variant = 'original'
                ):
        """

        :param input_endog: numpy array of intermittent demand time series
        :param forecast_length: forecast horizon
        :param croston_variant: croston model type
        :return: dictionary of model parameters, in-sample forecast, and out-of-sample forecast
        """
        
        global input_series1
        input_series = np.asarray(input_endog)
        input_series1 = input_series
        global epsilon1
        epsilon = 1e-7
        epsilon1 = epsilon
        input_length = len(input_series)
        nzd = np.where(input_series != 0)[0]
        
        if list(nzd) != [0]:
                
                try:
                    w_opt = _croston_opt(
                                            input_series = input_series,
                                            input_series_length = input_length,
                                            croston_variant = croston_variant,
                                            epsilon = epsilon,                                            
                                            w = None,
                                            nop = 1
                                        )
                    
                    croston_training_result = _croston(
                                                            input_series = input_series, 
                                                            input_series_length = input_length,
                                                            croston_variant = croston_variant,
                                                            w = w_opt, 
                                                            h = forecast_length,
                                                            epsilon = epsilon,
                                                      )
                    croston_model = croston_training_result['model']
                    croston_fittedvalues = croston_training_result['in_sample_forecast']
                    
                    croston_forecast = croston_training_result['out_of_sample_forecast']

                except Exception as e:
                    
                    croston_model = None
                    croston_fittedvalues = None
                    croston_forecast = None
                    print(str(e))
        
        else:
            
            croston_model = None
            croston_fittedvalues = None
            croston_forecast = None        
        
        
        return {
                    'croston_model':            croston_model,
                    'croston_fittedvalues':     croston_fittedvalues,
                    'croston_forecast':         croston_forecast
                }

def _croston(
                 input_series, 
                 input_series_length,
                 croston_variant,
                 w, 
                 h,                  
                 epsilon
             ):
    
    # Croston decomposition
    nzd = np.where(input_series != 0)[0] # find location of non-zero demand
    
    k = len(nzd)
    z = input_series[nzd] # demand
    
    x = np.concatenate([[nzd[0]], np.diff(nzd)]) # intervals

    # initialize
    
    init = [z[0], np.mean(x)]
    
    zfit = np.array([None] * k)
    xfit = np.array([None] * k)

    # assign initial values and prameters
    
    zfit[0] = init[0]
    xfit[0] = init[1]

    if len(w) == 1:
        a_demand = w[0]
        a_interval = w[0]
    
    else:
        a_demand = w[0]
        a_interval = w[1]
    
    # compute croston variant correction factors
    #   sba: syntetos-boylan approximation
    #   sbj: shale-boylan-johnston
    #   tsb: teunter-syntetos-babai        
    
    if croston_variant == 'sba':
        correction_factor = 1 - (a_interval / 2)
    
    elif croston_variant == 'sbj':
        correction_factor = (1 - a_interval / (2 - a_interval + epsilon))
        
    else:
        correction_factor = 1
    
    # fit model
    
    for i in range(1,k):
        zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1]) # demand
        xfit[i] = xfit[i-1] + a_interval * (x[i] - xfit[i-1]) # interval
        
    cc = correction_factor * zfit / (xfit + epsilon)
    
    croston_model = {
                        'a_demand':             a_demand,
                        'a_interval':           a_interval,
                        'demand_series':        pd.Series(zfit),
                        'interval_series':      pd.Series(xfit),
                        'demand_process':       pd.Series(cc),
                        'correction_factor':    correction_factor
                    }
    
    # calculate in-sample demand rate
    
    frc_in = np.zeros(input_series_length)
    tv = np.concatenate([nzd, [input_series_length]]) # Time vector used to create frc_in forecasts
    
    for i in range(k):
        frc_in[tv[i]:min(tv[i+1], input_series_length)] = cc[i]

    # forecast out_of_sample demand rate
    
    # croston 中的weight符合指数分布，因此并越到后面，weight下降越快。
    # 因此， croston的最后一个值，基本上是等于最后一个fitted value。
    if h > 0:
        frc_out = np.array([cc[k-1]] * h)
    else:
        frc_out = None
    
    return_dictionary = {
                            'model':                    croston_model,
                            'in_sample_forecast':       frc_in,
                            'out_of_sample_forecast':   frc_out
                        }
    
    return return_dictionary

def _croston_opt(
                    input_series, 
                    input_series_length, 
                    croston_variant,
                    epsilon,
                    w = None,
                    nop = 1
                ):
    
    p0 = np.array([0.1] * nop)

    # # 通过minimize的方式，获取到一个最优化值。
    # # 感觉可以深挖下这个的算法耶。。里面还含有分布函数的选择。
    # wopt = minimize(
    #                     fun = _croston_cost, 
    #                     x0 = p0, 
    #                     method='Nelder-Mead',
    #                     args=(input_series, input_series_length, croston_variant, epsilon)
    #                 )

    # Bounded region of parameter space
    # 为相同值时
    # pbounds = {'p0': (0, 1)}

    # 为不同值时
    pbounds = {'p': (0 ,1), 'q': (0, 1)}

    optimizer = BayesianOptimization(
        # f=calc_opt1,
        f=calc_opt2,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize()

    wopt = optimizer.max

    # constrained_wopt = np.minimum([1], np.maximum([0], wopt['params']['p0']))
    constrained_wopt = [wopt['params']['p'], wopt['params']['q']]
    return constrained_wopt
    

def _croston_cost(
                    p0,
                    input_series,
                    input_series_length,
                    croston_variant,
                    epsilon
                ):
    
    # cost function for croston and variants
    
    frc_in = _croston(
                            input_series = input_series,
                            input_series_length = input_series_length,
                            croston_variant = croston_variant,
                            w=p0,
                            h=0,
                            epsilon = epsilon
                        )['in_sample_forecast']
        
    E = input_series - frc_in
    E = E[E != np.array(None)]
    E = np.mean(E ** 2)

    if len(p0) < 2:
        print(('demand : {0}  a_interval: {1} rmse: {2}').format(p0[0], p0[0], E))
    else:
        print(('demand : {0}  a_interval: {1} rmse: {2}').format(p0[0], p0[1], E))

    return E

def calc_opt1(p0):
    E = _croston_cost([p0],input_series1, len(input_series1),'original',epsilon1)
    return -E

def calc_opt2(p, q):
    E = _croston_cost([p,q],input_series1, len(input_series1),'original',epsilon1)
    return -E


# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]
# ts = np.insert(a, idxs, val)


input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
input_data = input_data.fillna(0)
ts = input_data['Feature']

fit_pred = fit_croston(ts, 4, 'original') # croston's method

# fit_pred = fit_croston(ts, 4, 'sba') # Syntetos-Boylan approximation
# fit_pred = fit_croston(ts, 4, 'sbj') # Shale-Boylan-Johnston


yhat = np.concatenate([fit_pred['croston_fittedvalues'], fit_pred['croston_forecast']])

opt_model = fit_pred['croston_model']
print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))

# print(ts)
# print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()

# |  28       | -6.613e+0 |  0.3492   |
# demand : 0.3498813038589822  a_interval: 0.3498813038589822 rmse: 6612838.035590242

# 为不同值时
# demand : 0.9021569555343326  a_interval: 0.1878818737137048 rmse: 5538580.716017791



# # Test
# W = [fit_pred['croston_model']['a_demand'], fit_pred['croston_model']['a_interval']]
# test_data = pd.read_csv("./data/M4DataSet/NewYearlyTest.csv")
# test_data = test_data.fillna(0)
# ts_test = test_data['Feature']

# test_out = _croston(ts_test, len(ts_test),'original',W,0, epsilon1)
# test_out = test_out['in_sample_forecast']

# E = test_out - ts_test
# E = E[E != np.array(None)]
# E = np.mean(E ** 2)
# print(('out: a_demand : {0}  a_interval: {1} rmse: {2}').format(W[0], W[1], E))

# # print(ts_test)
# # print(test_out)

# plt.plot(ts_test)
# plt.plot(test_out)

# plt.show()

# # print(test_rmse)

# # out: a_demand : 0.3495037841339784  a_interval: 0.3495037841339784 rmse: 13015818.528248474
