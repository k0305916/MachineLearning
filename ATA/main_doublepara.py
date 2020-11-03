import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# p, q 看一下是否可以通过动态获取。
def fit_ata(
                    input_endog,
                    forecast_length               
                ):
        """
        :param input_endog: numpy array of intermittent demand time series
        :param forecast_length: forecast horizon
        :return: dictionary of model parameters, in-sample forecast, and out-of-sample forecast
        """
        input_series = np.asarray(input_endog)
        epsilon = 1e-7
        input_length = len(input_series)
        nzd = np.where(input_series != 0)[0]
        
        if list(nzd) != [0]:
                
                try:
                    w_opt = _ata_opt(
                                            input_series = input_series,
                                            input_series_length = input_length,
                                            epsilon = epsilon,                                            
                                            w = None,
                                            nop = 2
                                        )
                    
                    ata_training_result = _ata(
                                                            input_series = input_series, 
                                                            input_series_length = input_length,
                                                            w = w_opt, 
                                                            h = forecast_length,
                                                            epsilon = epsilon,
                                                      )
                    ata_model = ata_training_result['model']
                    ata_fittedvalues = ata_training_result['in_sample_forecast']
                    
                    ata_forecast = ata_training_result['out_of_sample_forecast']

                    ata_demand_series = ata_training_result['fit_output']

                except Exception as e:
                    
                    ata_model = None
                    ata_fittedvalues = None
                    ata_forecast = None
                    print(str(e))
        
        else:
            
            ata_model = None
            ata_fittedvalues = None
            ata_forecast = None        
        
        
        return {
                    'ata_model':            ata_model,
                    'ata_fittedvalues':     ata_fittedvalues,
                    'ata_forecast':         ata_forecast,
                    'ata_demand_series':    ata_demand_series
                }

def _ata(
            input_series, 
            input_series_length,
            w, 
            h,                  
            epsilon
            ):
    
    # ata decomposition
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

    correction_factor = 1
    
    p = w[0]
    q = w[1]
    # fit model
    for i in range(1,k):
        a_demand = p / nzd[i]
        a_interval = q / nzd[i]

        # 这个地方其实有些不懂, 为什么要这样操作？
        if nzd[i] <= p:
            zfit[i] = z[i]
        else:
            zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1]) # demand


        if nzd[i] <= q:
            xfit[i] = z[i] - z[i-1]
        else:
            xfit[i] = xfit[i-1] + a_interval * (x[i] - xfit[i-1]) # interval
            
        
    cc = correction_factor * zfit / (xfit + epsilon)
    
    ata_model = {
                        'a_demand':             p,
                        'a_interval':           q,
                        'demand_series':        pd.Series(zfit),
                        'interval_series':      pd.Series(xfit),
                        'demand_process':       pd.Series(cc),
                        'correction_factor':    correction_factor
                    }
    
    # calculate in-sample demand rate
    
    frc_in = np.zeros(input_series_length)
    tv = np.concatenate([nzd, [input_series_length]]) # Time vector used to create frc_in forecasts

    zfit_output = np.zeros(input_series_length)
    
    for i in range(k):
        frc_in[tv[i]:min(tv[i+1], input_series_length)] = cc[i]
        zfit_output[tv[i]:min(tv[i+1], input_series_length)] = zfit[i]

    # forecast out_of_sample demand rate
    
    # ata 中的weight符合超几何分布，因此并不会出现越到后面，weight下降越快。
    # 因此， ata的最后一个值，并不会100%等于最后一个fitted value。
    # 从forecast的公式可知，其中并没有 h 可迭代参数，因此，forecast的结果都是最后一个。
    if h > 0:
        frc_out = np.array([cc[k-1]] * h)
    else:
        frc_out = None
    
    return_dictionary = {
                            'model':                    ata_model,
                            'in_sample_forecast':       frc_in,
                            'out_of_sample_forecast':   frc_out,
                            'fit_output':               zfit_output
                        }
    
    return return_dictionary

def _ata_opt(
                    input_series, 
                    input_series_length, 
                    epsilon,
                    w = None,
                    nop = 2
                ):

    p0 = np.array([1,1])
    pbounds = ((1, input_series_length), (0, input_series_length))

    # 通过minimize的方式，获取到一个最优化值。
    # 感觉可以深挖下这个的算法耶。。里面还含有分布函数的选择。
    wopt = minimize(
                        fun = _ata_cost, 
                        x0 = p0, 
                        method='L-BFGS-B',
                        bounds=pbounds,
                        args=(input_series, input_series_length, epsilon)
                    )
    
    # constrained_wopt = np.minimum([1], np.maximum([0], wopt.x))
    constrained_wopt = wopt.x
    
    return constrained_wopt

# 仅仅通过rmse来判断，容易产生过拟合的问题，因此需要添加新的正则化来减轻过拟合~
def _ata_cost(
                p0,
                input_series,
                input_series_length,
                epsilon
                ):
    frc_in = _ata(
                    input_series = input_series,
                    input_series_length = input_series_length,
                    w=p0,
                    h=0,
                    epsilon = epsilon
                        )['in_sample_forecast']
    E = input_series - frc_in
    E = E[E != np.array(None)]
    E = np.mean(E ** 2)

    # print(("count: {0}  p: {1}  q: {2}  E: {3}").format(count, p0[0], p0[1], E))
    print(("p: {0}  q: {1}  E: {2}").format(p0[0], p0[1], E))
    # count = count + 1
    return E

def ata_forecast(w, init, h, epsilon):
    z = np.array([None] * h)
    zfit = np.array([None] * h)
    xfit = np.array([None] * h)

    # assign initial values and prameters
    
    z[0] = init
    zfit[0] = z[0]
    xfit[0] = 1

    p = w[0]
    q = w[1]
    # fit model
    for i in range(1,h):
        a_demand = p / i
        a_interval = q / i

        if i <= p:
            z[i] = z[i-1]
            zfit[i] = z[i]
        else:
            zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1]) # demand


        if i <= q:
            xfit[i] = z[i] - z[i-1]
        else:
            xfit[i] = xfit[i-1] + a_interval * (i - xfit[i-1]) # interval
            
        
    cc =  zfit / (xfit + epsilon)

    return cc

# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]
# ts = np.insert(a, idxs, val)


input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
input_data = input_data.fillna(0)
ts = input_data['Feature']

fit_pred = fit_ata(ts, 0)


# yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])
yhat = fit_pred['ata_demand_series']

opt_model = fit_pred['ata_model']
print("opt P: {0}   Q: {1}".format(opt_model["p"],opt_model["q"]))
# print(ts)
# print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()

# p: 13887.31775824237  q: 2157.2851580033325  E: 5840921.397489025

# # -----------------------Invalid-------------------------------
# # Test
# init = fit_pred['ata_forecast'][-1]
# # init = 479
# W = [fit_pred['ata_model']['a_demand'], fit_pred['ata_model']['a_interval']]
# # W = [13887.31775824237, 2157.2851580033325]
# test_data = pd.read_csv("./data/M4DataSet/NewYearlyTest.csv")
# test_data = test_data.fillna(0)
# ts_test = test_data['Feature']

# test_out = ata_forecast(W, init, len(ts_test), 1e-7)

# # E = test_out - ts_test
# # E = E[E != np.array(None)]
# # E = np.mean(E ** 2)
# # print(('out: a_demand : {0}  a_interval: {1} rmse: {2}').format(W[0], W[1], E))

# # print(ts_test)
# # print(test_out)

# plt.plot(ts_test)
# plt.plot(test_out)

# plt.show()
# # ----------------------------------------------------------