import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd

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
    
    zfit = np.array([None] * input_series_length)
    xfit = np.array([0] * input_series_length)
    
    if len(w) < 2:
        p = w[0]
        q = w[0]
    else:
        p = w[0]
        q = w[1]


    if q > p:
        print("error")

    # fit model
    cc = []
#     cc.append(input_series[0])
    j = 1
    k = 1
    
    xxx = []
    count = 1
    for tmp in input_series:
            xxx.append(count)
            if tmp != 0:
                count = 1
            else:
                count += 1
    
    for i in range(len(input_series)):
        if(input_series[i] == 0):
            k = k + 1
#             z1 = z0
#             n1 = n0
#         else:
        if(i <= p):
            z0 = input_series[i]
        else:
            z1 = p/i*input_series[i]+ (1-p/i)*z0
            z0 = z1

        if(i <= q):
            if (i == 0):
                n0 = 0
            else:
#                 n0 = input_series[i] - input_series[i-1]
                n0 = k
#                     n0 = xxx[i]
        else:
#                 n1 =  q/i*xxx[i] + (1 - q/i)*n0
            n1 =  q/i*k + (1 - q/i)*n0
            n0 = n1

        if(input_series[i] != 0):
            k = 1

        zfit[i] = z0
        xfit[i] = n0
    
    
#     for i in range(0,input_series_length):
#         a_demand = p / j
#         a_interval = q / j
        
#         if input_series[i] == 0:
#             k += 1

# #         if input_series[i] == 0:
# #             zfit[i] = zfit[i-1]
# #             xfit[i] = xfit[i-1]
# #         else:
#         if i <= p:
#             zfit[i] = input_series[i]
#         else:
#             zfit[i] = a_demand * input_series[i] + (1 - a_demand) * zfit[i-1]

#         if i == 0:
#             xfit[i] = 0
#         elif i <= q:
#             xfit[i] = input_series[i] - input_series[i-1]
#         else:
#             xfit[i] = a_interval * k + (1-a_interval) * xfit[i-1]

        if xfit[i] == 0:
            cc.append(zfit[i])
        else:
            cc.append(zfit[i] / (xfit[i]+1e-7))
        j+=1
        
#         if input_series[i] != 0:
#             k = 1

    print(("p: {0} q: {1} last interval: {2}").format(p, q, xfit[-1]))
           
    ata_model = {
                        'a_demand':             p,
                        'a_interval':           q,
                        'demand_series':        pd.Series(zfit),
                        'interval_series':      pd.Series(xfit),
                        'demand_process':       pd.Series(cc)
                    }
    
    # calculate in-sample demand rate
    frc_in = cc

    # forecast out_of_sample demand rate
    
    # ata 中的weight符合超几何分布，因此并不会出现越到后面，weight下降越快。
    # 因此， ata的最后一个值，并不会100%等于最后一个fitted value。
    # 从forecast的公式可知，其中并没有 h 可迭代参数，因此，forecast的结果都是最后一个。
    # if h > 0:
    #     frc_out = np.array([cc[k-1]] * h)
    # else:
    #     frc_out = None

    # f = open("frc_in.txt","tw")
    # for i in range(len(zfit_output)) :
        # f.write(str(zfit_output[i]) +'\t' + str(xfit_output[i]) + '\t' + str(frc_in[i]) + '\n')
    

    if h > 0:
        frc_out = []
        a_demand = p / input_series_length
        a_interval = q / input_series_length
        zfit_frcout = a_demand * input_series[-1] + (1-a_demand)*zfit[-1]
        xfit_frcout = a_interval * xxx[-1] + (1-a_interval)*xfit[-1]


        for i in range(1,h+1):
            result = zfit_frcout / xfit_frcout
            frc_out.append(result)
            # f.write(str(zfit_frcout) +'\t' + str(xfit_frcout) + '\t' + str(result) + '\n')
    else:
        frc_out = None

    # f.close()
    return_dictionary = {
                            'model':                    ata_model,
                            'in_sample_forecast':       frc_in,
                            'out_of_sample_forecast':   frc_out,
                            'fit_output':               cc
                        }
    
    return return_dictionary

def _ata_opt(
                    input_series, 
                    input_series_length, 
                    epsilon,
                    w = None,
                    nop = 2
                ):

    # # 0.2 * N
    # init_p = np.random.randint(1, input_series_length)
    # init_q = np.random.randint(0, init_p)
    # p0 = np.array([init_p,init_q])
    p0 = np.array([1.0,0.0])
    pbounds = ((1, input_series_length), (0, input_series_length))


    # # p0 = np.array([1])
    # # pbounds = ((1, input_series_length),)

    # # # # 通过minimize的方式，获取到一个最优化值。
    # # # # 感觉可以深挖下这个的算法耶。。里面还含有分布函数的选择。
    # # # # 传入梯度下降的公式，则可以降低计算的消耗。。。。
    # # # # 调整步长，来修正梯度下降的效率？
    wopt = minimize(
                        fun = _ata_cost, 
                        x0 = p0, 
                        method='L-BFGS-B',
                        bounds=pbounds,
                        args=(input_series, input_series_length, epsilon)
                    )

    constrained_wopt = wopt.x
    fun = wopt.fun


    # # p0 = np.array([1])

    # # wopt = minimize(
    # #                     fun = _ata_cost, 
    # #                     x0 = p0, 
    # #                     method='Nelder-Mead',
    # #                     args=(input_series, input_series_length, epsilon)
    # #                 )

    # constrained_wopt = wopt.x
    # fun = wopt.fun


    # pbounds = ((1, input_series_length), (0, input_series_length))
    # wopt = scipy.optimize.brute(_ata_cost,pbounds,
    #                             args=(input_series, input_series_length, epsilon))
    
    # # constrained_wopt = np.minimum([1], np.maximum([0], wopt.x))
    # constrained_wopt = wopt
    # fun = 0

    # # 双重退火
    # pbounds = ((1, input_series_length), (0, input_series_length))
    # wopt = scipy.optimize.dual_annealing(_ata_cost, pbounds,
    #                               args=(input_series, input_series_length, epsilon))

    # constrained_wopt = wopt.x
    # fun = wopt.fun

    # # 简单同源全局优化
    # pbounds = ((1, input_series_length), (0, input_series_length))
    # wopt = scipy.optimize.shgo(_ata_cost, pbounds,
    #                               args=(input_series, input_series_length, epsilon))

    # constrained_wopt = wopt.x
    # fun = wopt.fun
    
    return (constrained_wopt, fun)

# 仅仅通过rmse来判断，容易产生过拟合的问题，因此需要添加新的正则化来减轻过拟合~
def _ata_cost(
                p0,
                input_series,
                input_series_length,
                epsilon
                ):
    # #防止进入负数区间
    if p0[0] <= 0 or p0[1] < 0:
        return 3.402823466E+38

    if p0[1] > p0[0]:
        return 3.402823466E+38
    
    output = _ata(
                    input_series = input_series,
                    input_series_length = input_series_length,
                    w=p0,
                    h=0,
                    epsilon = epsilon
                        )

    # MSE： 在该算法中，optimize时，MSE（RMSE）并不是一个好的选择。-------------------------------------
#     frc_in.pop()
    frc_in = output['fit_output']
    
#     frc_in = output['in_sample_forecast']
    intv_in = output['model']['interval_series'].values

    
    xxx = []
    count = 1
    for tmp in input_series:
            xxx.append(count)
            if tmp != 0:
                count = 1
            else:
                count += 1

#     E = np.array(input_series)/np.array(xxx) - np.array(frc_in)
#     E = E[E != np.array(None)]
#     E = np.sqrt(np.mean(E ** 2))
    
    Ev = xxx - intv_in
    Ev = Ev[Ev != np.array(None)]
    Ev = np.sqrt(np.mean(Ev ** 2))
# #     print(E, Ev)
    E = Ev
    
#     E = 2/(1000/(E+1.0e-8) + 1/(Ev + 1.0e-8))
#     print(xxx)
#     print(frc_in)
                
#     E = xxx - frc_in
#     print('residual: ', end='')
#     for i in E:
#         print(i,end=',')

    # # 变形MSE
    # # count = min(input_series_length-1,(int)(p0[0]))
    # # indata = input_series[count:]
    # # outdata = frc_in[count:]
    # # E = indata - outdata
    
    # standard MSE

    
    

    # # standard RMSE
    # E = E[E != np.array(None)]
    # E = np.sqrt(np.mean(E ** 2))

    # # # # MAPE 针对非0数据的话，效果会比较好--------------------------------
    # E = np.abs((frc_in - input_series)) / (np.abs(input_series) + 1e-7)
    # E = E.sum() / input_series_length

    # # PAL MAPE
    # E = np.abs((frc_in - input_series))
    # up = E.sum()
    # low = input_series.sum()
    # E = up / low

    # print(("count: {0}  p: {1}  q: {2}  E: {3}").format(count, p0[0], p0[1], E))
    if len(p0) < 2:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[0], E))
    else:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[1], E))
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
# yhat = fit_pred['ata_demand_series']

# # excel dataset
def my_data_cont():
    ts = [
        362.35, 361.51, 363.51, 362.56, 361.88, 361.63, 361.35, 362.82, 360.64, 362.35, 362.77, 361.79, 
        361.41, 360.35, 357.75, 356.11, 355.24, 353.58, 347.49, 333.02, 333.50, 331.26, 322.03, 314.66, 
        312.74, 307.52, 304.87, 301.73, 300.62, 303.82, 307.40, 309.85, 311.06, 312.83, 314.97, 318.79, 
        320.81, 323.62, 325.66, 331.44, 332.61, 335.44, 336.57, 338.26, 337.20, 338.30, 342.17, 342.40, 
        341.63, 344.27, 342.50, 343.39, 343.57, 346.44, 347.07, 347.47, 349.21, 349.56, 352.03, 353.95, 
        355.17, 352.91, 356.45, 358.93, 362.35, 361.51, 363.51, 362.56, 361.88, 361.63, 361.35, 362.82, 
        360.64, 362.35, 362.77, 322.03, 314.66, 312.74, 307.52, 304.87, 301.73, 300.62
    ]
    return ts

# 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,

def ata(x, p, q):
    fitted = list()
    cc = 1
    if(q > p):
        error('q cannot be larger than p')
    for i in range(len(x)):
        if(x[i] == 0):
            cc = cc + 1

        if(i <= p):
            z0 = x[i]
        else:
            z1 = p/i*x[i]+ (1-p/i)*z0
            z0 = z1

        if(i <= q):
            if (i == 0):
                n0 = 0
            else:
                n0 = x[i] - x[i-1]
        else:
            n1 =  q/i*cc + (1 - q/i)*n0
            n0 = n1

        if(x[i] != 0):
            cc = 1
        fitted.append(z0)
    return fitted, z1, n1

def my_data():
    ts = [
    -11617.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-11617.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,-11617.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,-12357.6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6]
    
    return ts

def crost(x, alpha = 0.1):
    z0 = x[0]
    q = 1
    n0 = 1
    fitted = list()
    for i in range(len(x)):
        if x[i] == 0:
            q = q + 1
            z1 = z0
            n1 = n0
        else:
            z1 = z0 + alpha*(x[i] - z0)
            n1 = alpha*q + (1-alpha)*n0
            z0 = z1
            n0 = n1
            q = 1
        fitted.append(z0)
    return fitted,z1, n1
#_dataset = pd.read_csv("data/M4DataSet/Monthly-train.csv")
#ts = _dataset['V362'].values[:1000]

# z, n = ata(ts, 60, 4)
# z, n = ata(ts, 300, 280)
# print(n)

ts = my_data()

# # single process
# fit_pred = _ata(
#                 input_series = np.asarray(ts), 
#                 input_series_length = len(ts),
#                 w = (30,0), 
#                 h = 6,
#                 epsilon = 1e-7
#                 )
        

# yhat = np.concatenate([fit_pred['in_sample_forecast'], fit_pred['out_of_sample_forecast']])
# # yhat = fit_pred['ata_demand_series']

# opt_model = fit_pred['model']
# print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))


# optimize process
fit_pred = fit_ata(ts, 6) # ata's method

yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

opt_model = fit_pred['ata_model']
print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))

plt.plot(ts)
plt.plot(yhat)

plt.show()
print("")
