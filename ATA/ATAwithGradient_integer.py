import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
import random

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

        global p_gradient
        global q_gradient

        p_gradient = 0.0
        q_gradient = 0.0
        
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
                                                            gradient=(0,0)
                                                      )
                    ata_model = ata_training_result['model']
                    ata_fittedvalues = ata_training_result['in_sample_forecast']
                    
                    ata_forecast = ata_training_result['out_of_sample_forecast']

                    ata_demand_series = ata_training_result['fit_output']

                    ata_mse = w_opt[0]

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
            epsilon,
            gradient
            ):
    
    zfit = np.array([None] * input_series_length)
    xfit = np.array([0.0] * input_series_length)
    
    if len(w) < 2:
        raise Exception('This is the exception you expect to handle')
    else:
        p = w[0]
        q = w[1]

        p_gradient = gradient[0]
        q_gradient = gradient[1]


    # if q > p:
    #     return

    # fit model
    cc = []
    nt = []
    j = 1
    k = 1
    p_gradientlist = []
    q_gradientlist = []
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

        # if i > 0 :
        #     aa = 0
        #     bb = 0
        #     if ((k - xfit[i-1]) * q + j * xfit[i-1]) != 0.0 and input_series[i] != 0 :
        #         p_gradientlist.append((input_series[i] - zfit[i-1]) / ((k - xfit[i-1]) * q + j * xfit[i-1]))
        #         aa = (input_series[i] - zfit[i-1]) / ((k - xfit[i-1]) * q + j * xfit[i-1])
        #     if ((k - xfit[i-1]) * q + j * xfit[i-1]) != 0.0 and input_series[i] != 0 :
        #         q_gradientlist.append(((input_series[i] - zfit[i-1]) * p + j * zfit[i-1]) * (xfit[i-1] - k) / (((k - xfit[i-1]) * q + j * xfit[i-1]) ** 2))
        #         bb = ((input_series[i] - zfit[i-1]) * p + j * zfit[i-1]) * (xfit[i-1] - k) / (((k - xfit[i-1]) * q + j * xfit[i-1]) ** 2)

            # print(("----p_gradient: {0}   q_gradient: {1}").format(aa, bb))

    # p_gradientlist.pop(0)
    # q_gradientlist.pop(0)
    # p_gradient = sum(p_gradientlist)
    # q_gradient = sum(q_gradientlist)
    # print(("p: {0} q: {1} last interval: {2}").format(p, q, xfit[-1]))
           
    ata_model = {
                        'a_demand':             p,
                        'a_interval':           q,
                        'demand_series':        pd.Series(zfit),
                        'interval_series':      pd.Series(xfit),
                        'demand_process':       pd.Series(cc),
                        'in_sample_nt':         pd.Series(nt),
                        'last_interval':        xfit[-1],
                        'gredient':             (p_gradient, q_gradient)
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
                            'gredient':                 (p_gradient, q_gradient)
                        }
    
    return return_dictionary

def _ata_opt(
                    input_series, 
                    input_series_length, 
                    epsilon,
                    w = None,
                    nop = 2
                ):
    pbounds = ((1, input_series_length), (1, input_series_length))
    orderList = []
    for j in range(pbounds[1][0], pbounds[1][1]+1):
        for i in range(pbounds[0][0], pbounds[0][1]+1):
            orderList.append((i,j))

    errormin = 9999999999.99
    min_pq = (1,1)
    for tmp in orderList:
        error = _ata_cost(tmp,input_series,input_series_length,epsilon)
        if error < errormin:
            errormin = error
            min_pq = tmp

    constrained_wopt = min_pq
    fun = 0

    #---------------------------------------------------------
    # pbounds = ((1, input_series_length), (1, input_series_length))
    # wopt = scipy.optimize.linprog(_ata_cost,bounds = pbounds,
    #                             args=(input_series, input_series_length, epsilon))
    # constrained_wopt = wopt
    # fun = 0
    
    return (constrained_wopt, fun)


g_p_gradient = 0.0
g_q_gradient = 0.0
def _ata_cost(
                p0,
                input_series,
                input_series_length,
                epsilon
                ):

    p = p0[0]
    q = p0[1]
    if q > p:
        return 99999999999999.999
    
    global g_p_gradient
    global g_q_gradient
    # -------------------------------------------------------------
    ata_result = _ata(
                    input_series = input_series,
                    input_series_length = input_series_length,
                    w=p0,
                    h=0,
                    epsilon = epsilon,
                    gradient= (0, 0)
                    )
    frc_in = ata_result['in_sample_forecast']
    # print(frc_in)
    # (g_p_gradient, g_q_gradient) = ata_result['gredient']

    E = input_series - frc_in

    # standard RMSE
    E = E[E != np.array(None)]
    # E = np.sqrt(np.mean(E ** 2))
    E = np.mean(E ** 2)

    # print(("p_gradient: {0}   q_gradient: {1}").format(g_p_gradient, g_q_gradient))
    if len(p0) < 2:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], 0.0, E))
    else:
        print(('p : {0}  q: {1} mse: {2}').format(p0[0], p0[1], E))
    return E

def ClacGradient(p0, input_series,
                input_series_length,
                epsilon):
    return np.array((g_p_gradient, g_q_gradient))

# def CalcGradient_(p0, input_series, frc_in, interval_input_series, interval_in, time):
#     p = p0[0]
#     q = p0[1]
#     for i in range(1, len(input_series)):
#         x_t = input_series[i]
#         x_t_1_hat = frc_in[i]
#         n_t = interval_input_series[i]
#         n_t_1_hat = interval_in[i]
#         t = time[i]
        
#         p_gradient += (x_t - x_t_1_hat) / ((n_t - n_t_1_hat) * q + t * n_t_1_hat)
#         q_gradient += -((x_t - x_t_1_hat) * p + t * x_t_1_hat)*(n_t - n_t_1_hat) / ((n_t - n_t_1_hat) * q + t * n_t_1_hat)^2

#     print(("p_gradient: {0}   q_gradient: {1}").format(p_gradient, q_gradient))


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

# excel dataset
# ts = [
#     362.35, 361.51, 363.51, 362.56, 361.88, 361.63, 361.35, 362.82, 360.64, 362.35, 362.77, 361.79, 
#     361.41, 360.35, 357.75, 356.11, 355.24, 353.58, 347.49, 333.02, 0, 331.26, 322.03, 314.66, 
#     312.74, 307.52, 304.87, 301.73, 300.62, 303.82, 307.40, 309.85, 311.06, 312.83, 314.97, 318.79, 
#     320.81, 323.62, 325.66, 331.44, 332.61, 335.44, 336.57, 338.26, 337.20, 338.30, 342.17, 342.40, 
#     341.63, 344.27, 342.50, 343.39, 343.57, 346.44, 347.07, 347.47, 349.21, 349.56, 352.03, 353.95, 
#     355.17, 352.91, 356.45, 358.93, 362.35, 361.51, 363.51, 362.56, 361.88, 361.63, 361.35, 362.82, 
#     360.64, 362.35, 362.77, 322.03, 314.66, 312.74, 307.52, 304.87, 301.73, 300.62
# ]

# _ts = pd.Series(ts).fillna(0).values


# # zero dataset
# ts = [
#     0,0,0,0,0,0,0,0,0,0
# ]

# ts = [
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
#     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-12357.6]


# ts = [362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
#       0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
#       0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
#       0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
#       0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
#       0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
#       0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
#       312.74, 0.0, 0.0, 0.0, 0.0]

_dataset = pd.read_csv("data/M4DataSet/Monthly-train.csv")
ts = _dataset['V560'].fillna(0).values[:1000]

# z, n = ata(ts, 60, 4)
# z, n = ata(ts, 300, 280)
# print(n)

# # single process
# fit_pred = _ata(
#                 input_series = np.asarray(ts), 
#                 input_series_length = len(ts),
#                 w = (79,12), 
#                 h = 6,
#                 epsilon = 1e-7
#                 )
        

# yhat = np.concatenate([fit_pred['in_sample_forecast'], fit_pred['out_of_sample_forecast']])
# # yhat = fit_pred['ata_demand_series']

# opt_model = fit_pred['model']
# print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))


# optimize process
fit_pred = fit_ata(ts, 20) # ata's method

yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

opt_model = fit_pred['ata_model']
# print(opt_model.)
print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))

# plt.plot(EGlobe)
plt.plot(ts)
plt.plot(yhat)

plt.show()
print("")

# for i in range(64):
#     plt.figure(1)
#     ax = plt.subplot(8, 8, i+1)
#     ax.set_title('dddddddddddd',fontsize=12,color='r')
#     plt.plot(ts)
#     # plt.plot(yhat)

# ts = pd.Series([np.nan,np.nan,1,np.nan,np.nan])

# result = ts.isnull().all()
# print(result)




# # big data to test
# _dataset = pd.read_csv("data/M4DataSet/Monthly-train.csv")
# count = 64
# i = 0
# while True:
#     if i == count:
#         break

#     number = 'V' + str(random.randrange(2795))
#     # isNan = np.all(_dataset[number])
#     isNan = _dataset[number].isnull().all()
#     if isNan == True:
#         continue

#     ts = _dataset[number].fillna(0).values[:1000]


#     # optimize process
#     fit_pred = fit_ata(ts, 6) # ata's method

#     yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

#     opt_model = fit_pred['ata_model']
#     print("item: {0} opt P: {1}   Q: {2}  last interval: {3}".format(i, opt_model["a_demand"],opt_model["a_interval"], opt_model['last_interval']))

#     plt.figure(1)
#     ax = plt.subplot(8, 8, i+1)
#     ax.set_title("P:{0} Q:{1} interval:{2}".format(opt_model["a_demand"],opt_model["a_interval"], opt_model['last_interval']),fontsize=6,color='r')
#     plt.plot(ts)
#     plt.plot(yhat)
#     i += 1

# plt.show()
# print("")
