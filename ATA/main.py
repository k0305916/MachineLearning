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
        a_demand1 = p / nzd[i]
        a_demand2 = (nzd[i] - p) / nzd[i]
        a_interval1 = q / nzd[i]
        a_internval2 = (nzd[i] - q) / nzd[i]

        if nzd[i] <= p:
            z[i] = x[i]
        if nzd[i] <= q:
            x[i] = z[i] - z[i-1]

        zfit[i] = a_demand1 * z[i] + a_demand2 * zfit[i-1] # demand
        xfit[i] = a_interval1 * x[i] + a_internval2 * xfit[i-1] # interval
        
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

def _ata_opt(
                    input_series, 
                    input_series_length, 
                    epsilon,
                    w = None,
                    nop = 1
                ):
    
    # p0 = np.array([1,1] * nop)
            
    # wopt = minimize(
    #                     fun = _ata_cost, 
    #                     x0 = p0, 
    #                     method='Nelder-Mead',
    #                     args=(input_series, input_series_length, epsilon)
    #                 )
    # 由于ATA中weight的获取是与t相关的。。因此，就不能动态的获取了。
    wopt
    
    constrained_wopt = np.minimum([1], np.maximum([0], wopt.x))
    
    return constrained_wopt
    

def _ata_cost(
                    p0,
                    input_series,
                    input_series_length,
                    epsilon
                ):
    
    # cost function for ata and variants
    
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

    return E

a = np.zeros(7)
val = [1.0,4.0,5.0,3.0]
idxs = [1,2-1,6-2,7-3]

ts = np.insert(a, idxs, val)

fit_pred = fit_ata(ts, 4) # ata's method


yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

print(ts)
print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()