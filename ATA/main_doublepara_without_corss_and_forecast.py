import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.optimize
import matplotlib.pyplot as plt

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
                                                            w = w_opt[0], 
                                                            h = forecast_length,
                                                            epsilon = epsilon,
                                                      )
                    ata_model = ata_training_result['model']
                    ata_fittedvalues = ata_training_result['in_sample_forecast']
                    
                    ata_forecast = ata_training_result['out_of_sample_forecast']

                    ata_demand_series = ata_training_result['fit_output']

                    ata_mse = w_opt[1]

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
                    'ata_demand_series':    ata_demand_series,
                    'ata_mse':              ata_mse
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
    
    # initialize
    x = np.concatenate([[nzd[0]], np.diff(nzd)]) # intervals
    init = [z[0], np.mean(x)]
    
    zfit = np.array([None] * k)
    xfit = np.array([None] * k)

    # assign initial values and prameters
    
    zfit[0] = init[0]
    xfit[0] = init[1]

    correction_factor = 1
    
    if len(w) < 2:
        p = w[0]
        q = w[0]
    else:
        p = w[0]
        q = w[1]
    # fit model
    cc = []
    nzd = np.delete(nzd, 0)
    nzd = np.concatenate([nzd, [input_series_length]])
    # a_demand = p / input_series_length
    # a_interval = q / input_series_length
    for i in range(0,k):
        # if nzd[i] != 0:
        a_demand = p / nzd[i]
        a_interval = q / nzd[i]

        if i == 0:
            xfit[i] = 0

        # 提升效率的操作方式
        if nzd[i] <= p:
            zfit[i] = z[i]
        else:
            # zfit[i] = zfit[i-1] + a_demand * (z[i] - zfit[i-1]) # demand
            zfit[i] = a_demand * z[i] + (1-a_demand) * (zfit[i-1] + xfit[i-1])   #demand


        # for single exponential smoothing of ata
        if q == 0:
            xfit[i] = 0
        else:
            # for holt's linear trend method of ata
            if i != 0:
                if nzd[i] <= q:
                    xfit[i] = z[i] - z[i-1]
                else:
                    xfit[i] = a_interval * (zfit[i] - zfit[i-1]) + (1-a_interval) * xfit[i-1] # interval

        cc.append(zfit[i] + xfit[i])
        
    # cc = correction_factor * zfit / (xfit + epsilon)
    
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
    xfit_output = np.zeros(input_series_length)
    
    for i in range(k):
        frc_in[tv[i]:min(tv[i+1], input_series_length)] = cc[i]
        zfit_output[tv[i]:min(tv[i+1], input_series_length)] = zfit[i]
        xfit_output[tv[i]:min(tv[i+1], input_series_length)] = xfit[i]

    # forecast out_of_sample demand rate
    
    # ata 中的weight符合超几何分布，因此并不会出现越到后面，weight下降越快。
    # 因此， ata的最后一个值，并不会100%等于最后一个fitted value。
    # 从forecast的公式可知，其中并没有 h 可迭代参数，因此，forecast的结果都是最后一个。
    # if h > 0:
    #     frc_out = np.array([cc[k-1]] * h)
    # else:
    #     frc_out = None

    f = open("frc_in.txt","tw")
    for i in range(len(zfit_output)) :
        f.write(str(zfit_output[i]) +'\t' + str(xfit_output[i]) + '\t' + str(frc_in[i]) + '\n')
    

    if h > 0:
        frc_out = []
        a_demand = p / input_series_length
        a_interval = q / input_series_length
        zfit_frcout = a_demand * z[-1] + (1-a_demand)*(zfit_output[-1] + xfit_output[-1])
        xfit_frcout = a_interval * (zfit_frcout - zfit_output[-1]) + (1-a_interval)*xfit_output[-1]


        for i in range(1,h+1):
            result = zfit_frcout + i * xfit_frcout
            frc_out.append(result)
            f.write(str(zfit_frcout) +'\t' + str(xfit_frcout) + '\t' + str(result) + '\n')
    else:
        frc_out = None

    f.close()
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

    # 0.2 * N
    # init_p = np.random.randint(1, input_series_length)
    # init_q = np.random.randint(0, init_p)
    # p0 = np.array([init_p,init_q])
    # pbounds = ((1, input_series_length), (0, input_series_length))


    # p0 = np.array([1])
    # pbounds = ((1, input_series_length),)

    # # # 通过minimize的方式，获取到一个最优化值。
    # # # 感觉可以深挖下这个的算法耶。。里面还含有分布函数的选择。
    # # # 传入梯度下降的公式，则可以降低计算的消耗。。。。
    # # # 调整步长，来修正梯度下降的效率？
    # wopt = minimize(
    #                     fun = _ata_cost, 
    #                     x0 = p0, 
    #                     method='L-BFGS-B',
    #                     bounds=pbounds,
    #                     args=(input_series, input_series_length, epsilon)
    #                 )


    # # p0 = np.array([1])

    # # wopt = minimize(
    # #                     fun = _ata_cost, 
    # #                     x0 = p0, 
    # #                     method='Nelder-Mead',
    # #                     args=(input_series, input_series_length, epsilon)
    # #                 )

    # constrained_wopt = wopt.x
    # fun = wopt.fun


    pbounds = ((1, input_series_length), (0, input_series_length))
    wopt = scipy.optimize.brute(_ata_cost,pbounds,
                                args=(input_series, input_series_length, epsilon))
    
    # constrained_wopt = np.minimum([1], np.maximum([0], wopt.x))
    constrained_wopt = wopt
    fun = 0
    
    return (constrained_wopt, fun)

# 仅仅通过rmse来判断，容易产生过拟合的问题，因此需要添加新的正则化来减轻过拟合~
def _ata_cost(
                p0,
                input_series,
                input_series_length,
                epsilon
                ):
    # #防止进入负数区间
    # if p0[0] < 0 or p0[1] < 0:
    #     return 3.402823466E+38
    # # Q: [0, P]  P: [1,n]
    # if p0[1] > p0[0]:
    #     return 3.402823466E+38
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

    # print(("count: {0}  p: {1}  q: {2}  E: {3}").format(count, p0[0], p0[1], E))
    if len(p0) < 2:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[0], E))
    else:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[1], E))
    # count = count + 1
    return E

# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]
# ts = np.insert(a, idxs, val)


# # Yearly dataset
# input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
# input_data = input_data.fillna(0)
# ts = input_data['Feature']
# # ts = input_data['Feature'][:1000]

# fit_pred = fit_ata(ts, 4) # ata's method

# yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])
# # yhat = fit_pred['ata_demand_series']

# #excel dataset
ts = [
362.35, 361.51, 363.51, 362.56, 361.88, 361.63, 361.35, 362.82, 360.64, 362.35, 362.77, 361.79, 
361.41, 360.35, 357.75, 356.11, 355.24, 353.58, 347.49, 333.02, 335.56, 331.26, 322.03, 314.66, 
312.74, 307.52, 304.87, 301.73, 300.62, 303.82, 307.40, 309.85, 311.06, 312.83, 314.97, 318.79, 
320.81, 323.62, 325.66, 331.44, 332.61, 335.44, 336.57, 338.26, 337.20, 338.30, 342.17, 342.40, 
341.63, 344.27, 342.50, 343.39, 343.57, 346.44, 347.07, 347.47, 349.21, 349.56, 352.03, 353.95, 
355.17, 352.91, 356.45, 358.93, 362.35, 361.51, 363.51, 362.56, 361.88, 361.63, 361.35, 362.82, 
360.64, 362.35, 362.77, 322.03, 314.66, 312.74, 307.52, 304.87, 301.73, 300.62
]

fit_pred = _ata(
                input_series = np.asarray(ts), 
                input_series_length = len(ts),
                w = (30,0), 
                h = 6,
                epsilon = 1e-7
                )
        

yhat = np.concatenate([fit_pred['in_sample_forecast'], fit_pred['out_of_sample_forecast']])
# yhat = fit_pred['ata_demand_series']

opt_model = fit_pred['model']
print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))

plt.plot(ts)
plt.plot(yhat)

plt.show()
print("")
