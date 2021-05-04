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
            epsilon,
            ):
    
    zfit = np.array([None] * input_series_length)
    xfit = np.array([0.0] * input_series_length)
    
    if len(w) < 2:
        raise Exception('This is the exception you expect to handle')
    else:
        p = w[0]
        q = w[1]


    # if q > p:
    #     return

    # fit model
    cc = []
    nt = []
    j = 1
    k = 1
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
           
    ata_model = {
                        'a_demand':             p,
                        'a_interval':           q,
                        'demand_series':        pd.Series(zfit),
                        'interval_series':      pd.Series(xfit),
                        'demand_process':       pd.Series(cc),
                        'in_sample_nt':         pd.Series(nt),
                        'last_interval':        xfit[-1],
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
    pbounds = {'p': (1, input_series_length), 'q': (1, input_series_length)}

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


# input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
# input_data = input_data.fillna(0)
# # ts = input_data['Feature'][:4000]
# ts = input_data['Feature']


# ts = np.asarray([
#     -11617.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-11617.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6])

ts = np.asarray([362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
      0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
      0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
      0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
      0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
      0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
      0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
      312.74, 0.0, 0.0, 0.0, 0.0])

# _dataset = pd.read_csv("data/M4DataSet/Monthly-train.csv")
# ts = _dataset['V362'].fillna(0).values[:1000]

fit_pred = fit_ata(ts, 20) # ata's method


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