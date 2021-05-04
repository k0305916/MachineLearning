import math                         # 导入模块
import random                       # 导入模块
import pandas as pd                 # 导入模块 YouCans, XUPT
import numpy as np                  # 导入模块 numpy，并简写成 np
import time


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
    for i in range(0, input_series_length):
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
                zfit[i] = a_demand * input_series[i] + \
                    (1 - a_demand) * zfit[i-1]

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
        j += 1

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
    # print(xfit[-1])
    # calculate in-sample demand rate
    frc_in = cc

    # forecast out_of_sample demand rate
    if h > 0:
        frc_out = []
        a_demand = p / input_series_length
        a_interval = q / input_series_length
        zfit_frcout = a_demand * input_series[-1] + (1-a_demand)*zfit[-1]
        xfit_frcout = a_interval * nt[-1] + (1-a_interval)*xfit[-1]

        for i in range(1, h+1):
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

    # -------------------------------------------------------------
    ata_result = _ata(
        input_series=input_series,
        input_series_length=input_series_length,
        w=p0,
        h=0,
        epsilon=epsilon,
        gradient=(0, 0)
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


def ParameterSetting(input_series_length):
    cName = "funcOpt"           # 定义问题名称 YouCans, XUPT
    nVar = 2                    # 给定自变量数量，y=f(x1,..xn)
    xMin = [1, 1]               # 给定搜索空间的下限，x1_min,..xn_min
    # 给定搜索空间的上限，x1_max,..xn_max
    xMax = [input_series_length, input_series_length]

    tInitial = 100.0            # 设定初始退火温度(initial temperature)
    tFinal = 1                 # 设定终止退火温度(stop temperature)
    alfa = 0.9954              # 设定降温参数，T(k)=alfa*T(k-1)
    meanMarkov = 4            # Markov链长度，也即内循环运行次数
    scale = 0.5               # 定义搜索步长，可以设为固定值或逐渐缩小
    return cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale


# 模拟退火算法
def OptimizationSSA(input_series, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale):
    # ====== 初始化随机数发生器 ======
    randseed = random.randint(1, 100)
    random.seed(randseed)  # 随机数发生器设置种子，也可以设为指定整数

    # ====== 随机产生优化问题的初始解 ======
    xInitial = np.zeros((nVar))   # 初始化，创建数组
    for v in range(nVar):
        # xInitial[v] = random.uniform(xMin[v], xMax[v]) # 产生 [xMin, xMax] 范围的随机实数
        xInitial[v] = random.randint(xMin[v], xMax[v])  # 产生 [xMin, xMax] 范围的随机整数
    # 调用子函数 _ata_cost 计算当前解的目标函数值
    fxInitial = _ata_cost(xInitial, input_series, len(input_series), 10e-7)  # m(k)：惩罚因子，初值为 1

    # ====== 模拟退火算法初始化 ======
    xNew = np.zeros((nVar))         # 初始化，创建数组
    xNow = np.zeros((nVar))         # 初始化，创建数组
    xBest = np.zeros((nVar))        # 初始化，创建数组
    xNow[:] = xInitial[:]          # 初始化当前解，将初始解置为当前解
    xBest[:] = xInitial[:]          # 初始化最优解，将当前解置为最优解
    fxNow = fxInitial              # 将初始解的目标函数置为当前值
    fxBest = fxInitial              # 将当前解的目标函数置为最优值
    print('x_Initial:{:.6f},{:.6f},\tf(x_Initial):{:.6f}'.format(
        xInitial[0], xInitial[1], fxInitial))

    recordIter = []                 # 初始化，外循环次数
    recordFxNow = []                # 初始化，当前解的目标函数值
    recordFxBest = []               # 初始化，最佳解的目标函数值
    recordPBad = []                 # 初始化，劣质解的接受概率
    kIter = 0                       # 外循环迭代次数，温度状态数
    totalMar = 0                    # 总计 Markov 链长度
    totalImprove = 0                # fxBest 改善次数
    nMarkov = meanMarkov            # 固定长度 Markov链

    # ====== 开始模拟退火优化 ======
    # 外循环，直到当前温度达到终止温度时结束
    tNow = tInitial                 # 初始化当前温度(current temperature)
    while tNow >= tFinal:           # 外循环，直到当前温度达到终止温度时结束
        # 在当前温度下，进行充分次数(nMarkov)的状态转移以达到热平衡
        kBetter = 0                 # 获得优质解的次数
        kBadAccept = 0              # 接受劣质解的次数
        kBadRefuse = 0              # 拒绝劣质解的次数

        # ---内循环，循环次数为Markov链长度
        for k in range(nMarkov):    # 内循环，循环次数为Markov链长度
            totalMar += 1           # 总 Markov链长度计数器

            # ---产生新解
            # 产生新解：通过在当前解附近随机扰动而产生新解，新解必须在 [min,max] 范围内
            # 方案 1：只对 n元变量中的一个进行扰动，其它 n-1个变量保持不变
            xNew[:] = xNow[:]
            v = random.randint(0, nVar-1)   # 产生 [0,nVar-1]之间的随机数
            xNew[v] = round(xNow[v] + scale * (xMax[v]-xMin[v])
                            * random.normalvariate(0, 1))
            # 满足决策变量为整数，采用最简单的方案：产生的新解按照四舍五入取整
            # 保证新解在 [min,max] 范围内
            xNew[v] = max(min(xNew[v], xMax[v]), xMin[v])

            # ---计算目标函数和能量差
            # 调用子函数 _ata_cost 计算新解的目标函数值
            fxNew = _ata_cost(xNew, input_series, len(input_series), 10e-7)
            deltaE = fxNew - fxNow

            # ---按 Metropolis 准则接受新解
            # 接受判别：按照 Metropolis 准则决定是否接受新解
            if fxNew < fxNow:  # 更优解：如果新解的目标函数好于当前解，则接受新解
                accept = True
                kBetter += 1
            else:  # 容忍解：如果新解的目标函数比当前解差，则以一定概率接受新解
                pAccept = math.exp(-deltaE / tNow)  # 计算容忍解的状态迁移概率
                if pAccept > random.random():
                    accept = True  # 接受劣质解
                    kBadAccept += 1
                else:
                    accept = False  # 拒绝劣质解
                    kBadRefuse += 1

            # 保存新解
            if accept == True:  # 如果接受新解，则将新解保存为当前解
                xNow[:] = xNew[:]
                fxNow = fxNew
                if fxNew < fxBest:  # 如果新解的目标函数好于最优解，则将新解保存为最优解
                    fxBest = fxNew
                    xBest[:] = xNew[:]
                    totalImprove += 1
                    scale = scale*0.99  # 可变搜索步长，逐步减小搜索范围，提高搜索精度

        # ---内循环结束后的数据整理
        # 完成当前温度的搜索，保存数据和输出
        pBadAccept = kBadAccept / (kBadAccept + kBadRefuse)  # 劣质解的接受概率
        recordIter.append(kIter)  # 当前外循环次数
        recordFxNow.append(round(fxNow, 4))  # 当前解的目标函数值
        recordFxBest.append(round(fxBest, 4))  # 最佳解的目标函数值
        recordPBad.append(round(pBadAccept, 4))  # 最佳解的目标函数值

        if kIter % 10 == 0:                           # 模运算，商的余数
            print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'.
                  format(kIter, tNow, pBadAccept, fxBest))

        # 缓慢降温至新的温度，降温曲线：T(k)=alfa*T(k-1)
        tNow = tNow * alfa
        kIter = kIter + 1
        fxBest = _ata_cost(xBest, input_series, len(input_series), 10e-7)  # 由于迭代后惩罚因子增大，需随之重构增广目标函数
        # ====== 结束模拟退火过程 ======

    print('improve:{:d}'.format(totalImprove))
    return xBest, fxBest
    # return kIter, xBest, fxBest, fxNow, recordIter, recordFxNow, recordFxBest, recordPBad

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


[cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale] = ParameterSetting(len(ts))
time_start = time.perf_counter()
# optimize process
fit_pred = OptimizationSSA(np.asarray(ts), nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale) # ata's method
print (("{0} s").format(time.perf_counter() - time_start))
# yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

# opt_model = fit_pred['ata_model']
# print(opt_model.)
print("opt P: {0}   Q: {1} mse: {2}".format(fit_pred[0][0],fit_pred[0][1], fit_pred[1]))

# # plt.plot(EGlobe)
# plt.plot(ts)
# plt.plot(yhat)

# plt.show()
# print("")

# print(math.pow(0.01, 1/ 1000))