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

        # 提升效率的操作方式
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
    # if h > 0:
    #     frc_out = np.array([cc[k-1]] * h)
    # else:
    #     frc_out = None

    if h > 0:
        frc_out = []
        frc_xfit = np.zeros(h+1)
        frc_zfit = np.zeros(h+1)
        i=0
        for i in range(h):
            if i == 0 :
                frc_zfit[0] = zfit[-1]
                frc_xfit[0] = xfit[-1]
            frc_out.append(frc_zfit[i] / frc_xfit[i])
            a_demand = p / (input_series_length + i)
            a_interval = q / (input_series_length + i)
            # 1. 以0作为观测值；
            frc_zfit[i+1] = (1-a_demand) * frc_zfit[i]
            # 2. 以平均值作为观测值；
            # frc_zfit[i+1] = np.mean(z) + (1-a_demand) * frc_zfit[i]

            #frc_xfit一直以拟合值作计算。
            # frc_xfit[i+1] = (1-a_interval) * frc_xfit[i]
            frc_xfit[i+1] = frc_xfit[i] + (1 - frc_xfit[i])* a_interval

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

    # p0 = np.array([3,1])
    init_p = np.random.randint(1, input_series_length)
    init_q = np.random.randint(0, init_p)
    p0 = np.array([init_p,init_q])
    # print(p0)
    pbounds = ((1, input_series_length), (0, input_series_length))

    # 通过minimize的方式，获取到一个最优化值。
    # 感觉可以深挖下这个的算法耶。。里面还含有分布函数的选择。
    # 传入梯度下降的公式，则可以降低计算的消耗。。。。
    # 调整步长，来修正梯度下降的效率？
    wopt = minimize(
                        fun = _ata_cost, 
                        x0 = p0, 
                        method='L-BFGS-B',
                        bounds=pbounds,
                        args=(input_series, input_series_length, epsilon)
                    )

    constrained_wopt = wopt.x
    fun = wopt.fun

    # wopt = scipy.optimize.brute(_ata_cost,pbounds,
    #                             args=(input_series, input_series_length, epsilon))
    
    # # constrained_wopt = np.minimum([1], np.maximum([0], wopt.x))
    # constrained_wopt = wopt
    # fun = 0
    
    return (constrained_wopt, fun)

# 仅仅通过rmse来判断，容易产生过拟合的问题，因此需要添加新的正则化来减轻过拟合~
def _ata_cost(
                p0,
                input_series,
                input_series_length,
                epsilon
                ):
    #防止进入负数区间
    if p0[0] < 0 or p0[1] < 0:
        return 3.402823466E+38
    # Q: [0, P]  P: [1,n]
    if p0[1] > p0[0]:
        return 3.402823466E+38
    frc_in = _ata(
                    input_series = input_series,
                    input_series_length = input_series_length,
                    w=p0,
                    h=0,
                    epsilon = epsilon
                        )['in_sample_forecast']

    # MSE-------------------------------------
    # E = input_series - frc_in

    # # count = min(input_series_length-1,(int)(p0[0]))
    # # indata = input_series[count:]
    # # outdata = frc_in[count:]
    # # E = indata - outdata
    
    # E = E[E != np.array(None)]
    # # E = np.sqrt(np.mean(E ** 2))
    # E = np.mean(E ** 2)

    # MAPE--------------------------------
    E1 = (np.fabs(input_series - frc_in))
    E2 = (np.fabs(input_series) + np.fabs(frc_in)) / 2
    E = E1 / E2
    E = E.sum() / len(input_series)

    # print(("count: {0}  p: {1}  q: {2}  E: {3}").format(count, p0[0], p0[1], E))
    print(("p: {0}  q: {1}  E: {2}").format(p0[0], p0[1], E))
    # count = count + 1
    return E

# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]
# ts = np.insert(a, idxs, val)


input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
input_data = input_data.fillna(0)
ts = input_data['Feature']
# ts = input_data['Feature'][:1000]

#-------------Cross validation------------------------------------
cv_count = 5
repeatscale = 0.1
split_count = int(len(ts) / cv_count)
# split_count = int((1+repeatscale) * split_count + 0.5)
split_range = int(split_count * repeatscale +0.5)
dataend = len(ts)
opt_para = []
for i in range(cv_count):
    start = i * split_count - split_range
    end = (i+1) * split_count + split_range
    if start < 0:
        start = 0
    if end > dataend:
        end = dataend
    data = ts[start: end]
    fit_pred = fit_ata(data, 4)
    opt_model = fit_pred['ata_model']
    para = (opt_model["a_demand"], opt_model["a_interval"])
    # MSE = _ata_cost((opt_model["a_demand"], opt_model["a_interval"]), ts, len(ts), 1e-7)
    MSE = fit_pred['ata_mse']
    print("cv: {0}\topt P: {1}\tQ: {2}\tMSE: {3}".format(i, opt_model["a_demand"], opt_model["a_interval"], MSE))
    opt_para.append({"para":para, "MSE": MSE, 'fited': fit_pred})



#--------------------------Output the best paras----------------------------
opt_para = sorted(opt_para, key=lambda item: item['MSE'])
best_para = opt_para[0]
test_data = np.asarray(ts)
fit_pred = _ata(test_data,len(test_data),best_para['para'],4,1e-7)

test_fited = fit_pred['in_sample_forecast']
E = test_data - test_fited
E = E[E != np.array(None)]
MSE = np.mean(E ** 2)

yhat = np.concatenate([fit_pred['in_sample_forecast'], fit_pred['out_of_sample_forecast']])
# yhat = fit_pred['ata_demand_series']

opt_model = fit_pred['model']
print("output P: {0}\tQ: {1}\tmse: {2}".format(opt_model["a_demand"],opt_model["a_interval"], MSE))
# print(ts)
# print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()
print("")


# round 1: output P: 969.6244064590888     Q: 217.45943750155297   mse: 7836237.429606486
# p: 13782.98556931908  q: 2499.000000000012  E: 2429.2208993823615 grid search
# p: 13887.31775824237  q: 2157.2851580033325  E: 5840921.397489025
# p: 14293.167068707324  q: 1e-08  E: 2687.3898919142553

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
