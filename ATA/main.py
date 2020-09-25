import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# p, q 看一下是否可以通过动态获取。
def fit_ata(
                    input_endog,
                    forecast_length,
                    p,
                    q                    
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
                    # w_opt = _ata_opt(
                    #                         input_series = input_series,
                    #                         input_series_length = input_length,
                    #                         epsilon = epsilon,                                            
                    #                         w = None,
                    #                         nop = 1
                    #                     )
                    
                    ata_training_result = _ata(
                                                            input_series = input_series, 
                                                            input_series_length = input_length,
                                                            w = [p,q], 
                                                            h = forecast_length,
                                                            epsilon = epsilon,
                                                      )
                    ata_model = ata_training_result['model']
                    ata_fittedvalues = ata_training_result['in_sample_forecast']
                    
                    ata_forecast = ata_training_result['out_of_sample_forecast']

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
                    'ata_forecast':         ata_forecast
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
                            'out_of_sample_forecast':   frc_out
                        }
    
    return return_dictionary

# 仅仅通过rmse来判断，容易产生过拟合的问题，因此需要添加新的正则化来减轻过拟合~
def _ata_cost(
                input_series,
                frc_in
                ):
    E = input_series - frc_in
    E = E[E != np.array(None)]
    E = np.mean(E ** 2)

    return E

a = np.zeros(7)
val = [1.0,4.0,5.0,3.0]
idxs = [1,2-1,6-2,7-3]

ts = np.insert(a, idxs, val)

min_rmse = 99999999
min_p = 0
min_q = 0

# 如何最优化p,q的过程~  重点~
for p in range(1,7):
    for q in range(0,p+1):
        fit_pred = fit_ata(ts, 4, p, q) # ata's method

        frc_in = fit_pred['ata_fittedvalues']

        rmse = _ata_cost(ts, frc_in)

        if(rmse < min_rmse):
            min_rmse = rmse
            min_p = p
            min_q = q
        print(("rms: {0} p: {1} q: {2}").format(rmse,p,q))

print(("min_p: {0}  min_q: {1}").format(min_p,min_q))

fit_pred = fit_ata(ts, 4, min_p, min_q) # ata's method
yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

print(ts)
print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()