import numpy as np
import pandas as pd
# from scipy.optimize import minimize
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt

def fit_ata(
                    input_endog,
                    forecast_length,
                ):
        """

        :param input_endog: numpy array of intermittent demand time series
        :param forecast_length: forecast horizon
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
    
    for i in range(k):
        frc_in[tv[i]:min(tv[i+1], input_series_length)] = cc[i]

    # forecast out_of_sample demand rate
    
    # ata 中的weight符合指数分布，因此并越到后面，weight下降越快。
    # 因此， ata的最后一个值，基本上是等于最后一个fitted value。
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
                    nop = 2
                ):

# Bounded region of parameter space
    pbounds = {'p': (1, input_series_length), 'q': (0, input_series_length)}

    optimizer = BayesianOptimization(
        f=calc_opt,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize()

    wopt = optimizer.max

    constrained_wopt = [wopt['params']['p'], wopt['params']['q']]
    
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

    if len(p0) < 2:
        print(('demand : {0}  a_interval: {1} rmse: {2}').format(p0[0], p0[0], E))
    else:
        print(('demand : {0}  a_interval: {1} rmse: {2}').format(p0[0], p0[1], E))

    return E

def calc_opt(p,q):
    E = _ata_cost([p,q],input_series1, len(input_series1),epsilon1)
    return -E


# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]
# ts = np.insert(a, idxs, val)


input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
input_data = input_data.fillna(0)
ts = input_data['Feature']

fit_pred = fit_ata(ts, 4) # ata's method


yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

print(ts)
print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()

# demand : 11372.36833477203  a_interval: 2099.0495890911106 rmse: 5868603.9318896765

# Test
W = [fit_pred['ata_model']['a_demand'], fit_pred['ata_model']['a_interval']]
test_data = pd.read_csv("./data/M4DataSet/NewYearlyTest.csv")
test_data = test_data.fillna(0)
ts_test = test_data['Feature']

test_out = _ata(ts_test, len(ts_test),W,0, 1e-7)
test_out = test_out['in_sample_forecast']

E = test_out - ts_test
E = E[E != np.array(None)]
E = np.mean(E ** 2)
print(('out: a_demand : {0}  a_interval: {1} rmse: {2}').format(W[0], W[1], E))

# print(ts_test)
# print(test_out)

plt.plot(ts_test)
plt.plot(test_out)

plt.show()

# out: a_demand : 11372.36833477203  a_interval: 2099.0495890911106 rmse: 1.8484276491153072e+19
# standard out: a_demand : 22457.631597389674  a_interval: 0.0 rmse: 138.65686705572062