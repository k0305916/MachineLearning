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

                    ata_demand_series = ata_training_result['fit_output']

                except Exception as e:
                    
                    ata_model = None
                    ata_fittedvalues = None
                    ata_forecast = None
                    ata_demand_series = None
                    print(str(e))
        
        else:
            
            ata_model = None
            ata_fittedvalues = None
            ata_forecast = None        
            ata_demand_series = None
        
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
        zfit_output[tv[i]] = zfit[i]

    # forecast out_of_sample demand rate
    
    # ata 中的weight符合指数分布，因此并越到后面，weight下降越快。
    # 因此， ata的最后一个值，基本上是等于最后一个fitted value。
    if h > 0:
        frc_out = np.array([cc[k-1]] * h)
    else:
        frc_out = None

    # if h > 0:
    #     frc_out = []
    #     frc_xfit = np.zeros(h+1)
    #     frc_zfit = np.zeros(h+1)
    #     i=0
    #     for i in range(h):
    #         if i == 0 :
    #             frc_zfit[0] = zfit[-1]
    #             frc_xfit[0] = xfit[-1]
    #         frc_out.append(frc_zfit[i] / frc_xfit[i])
    #         a_demand = p / (input_series_length + i)
    #         a_interval = q / (input_series_length + i)
    #         # 1. 以0作为观测值；
    #         # frc_zfit[i+1] = (1-a_demand) * frc_zfit[i]
    #         # 2. 以平均值作为观测值；
    #         # frc_zfit[i+1] = np.mean(z) + (1-a_demand) * frc_zfit[i]
    #         frc_zfit[i+1] = np.sum(frc_zfit) / (i+1)  + (1-a_demand) * frc_zfit[i]
    #         # frc_zfit[i+1] = zfit[-1] * a_demand + (1-a_demand) * (frc_zfit[i] + i * xfit[-1])
    #         # frc_zfit[i+1] = frc_zfit[i] + a_demand * (zfit[-1] - frc_zfit[i])

    #         #frc_xfit一直以拟合值作计算。
    #         # frc_xfit[i+1] = (1-a_interval) * frc_xfit[i]
    #         frc_xfit[i+1] = frc_xfit[i] + (1 - frc_xfit[i])* a_interval
    #         # frc_xfit[i+1] = a_interval * (frc_zfit[i+1] - frc_zfit[i])  + (1 - a_interval) * (i * xfit[-1])
    #         # frc_xfit[i+1] = frc_xfit[i] + a_interval * (xfit[-1] - frc_xfit[i])

            
    # else:
    #     frc_out = None
    
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
    

# 可以尝试使用其他的statistic指标来进行计算: RMSE, MAPE等
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
    # MSE-------------------------------------
    E = input_series - frc_in

    # count = min(input_series_length-1,(int)(p0[0]))
    # indata = input_series[count:]
    # outdata = frc_in[count:]
    # E = indata - outdata
    
    E = E[E != np.array(None)]
    # E = np.sqrt(np.mean(E ** 2))
    E = np.mean(E ** 2)

    # # MAPE--------------------------------
    # E1 = (np.fabs(input_series - frc_in))
    # E2 = (np.fabs(input_series) + np.fabs(frc_in)) / 2
    # E = E1 / E2
    # E = E.sum() / len(input_series)


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
# ts = input_data['Feature'][:4000]
ts = input_data['Feature']

# demand : 1.0  a_interval: 0.0 rmse: 0.03419066029127796

fit_pred = fit_ata(ts, 4) # ata's method


yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])
# yhat = fit_pred['ata_demand_series']
opt_model = fit_pred['ata_model']
print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))

# print(ts)
# print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()

# demand : 11372.36833477203  a_interval: 2099.0495890911106 rmse: 5868603.9318896765

# # Test
# W = [fit_pred['ata_model']['a_demand'], fit_pred['ata_model']['a_interval']]
# test_data = pd.read_csv("./data/M4DataSet/NewYearlyTest.csv")
# test_data = test_data.fillna(0)
# ts_test = test_data['Feature']

# test_out = _ata(ts_test, len(ts_test),W,0, 1e-7)
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

# # out: a_demand : 11372.36833477203  a_interval: 2099.0495890911106 rmse: 1.8484276491153072e+19
# # standard out: a_demand : 22457.631597389674  a_interval: 0.0 rmse: 138.65686705572062