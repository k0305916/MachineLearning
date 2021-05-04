# 模拟退火算法 程序：求解线性规划问题（整数规划）
# Program: SimulatedAnnealing_v4.py
# Purpose: Simulated annealing algorithm for function optimization
# v4.0: 整数规划：满足决策变量的取值为整数（初值和新解都是随机生成的整数）
# Copyright 2021 YouCans, XUPT
# Crated：2021-05-01

#  -*- coding: utf-8 -*-
import math                         # 导入模块
import random                       # 导入模块
import pandas as pd                 # 导入模块 YouCans, XUPT
import numpy as np                  # 导入模块 numpy，并简写成 np
import matplotlib.pyplot as plt     
from datetime import datetime

#=== 关注 Youcans，分享更多原创系列 https://www.cnblogs.com/youcans/ ===
# 子程序：定义优化问题的目标函数
def cal_Energy(X, nVar, mk): 	# m(k)：惩罚因子，随迭代次数 k 逐渐增大
    p1 = (max(0, 6*X[0]+5*X[1]-60))**2
    p2 = (max(0, 10*X[0]+20*X[1]-150))**2
    fx = -(10*X[0]+9*X[1])
    return fx+mk*(p1+p2)


# 子程序：模拟退火算法的参数设置
def ParameterSetting():
    cName = "funcOpt"           # 定义问题名称 YouCans, XUPT
    nVar = 2                    # 给定自变量数量，y=f(x1,..xn)
    xMin = [0, 0]               # 给定搜索空间的下限，x1_min,..xn_min
    xMax = [8, 8]               # 给定搜索空间的上限，x1_max,..xn_max

    tInitial = 100.0            # 设定初始退火温度(initial temperature)
    tFinal  = 1                 # 设定终止退火温度(stop temperature)
    alfa    = 0.98              # 设定降温参数，T(k)=alfa*T(k-1)
    meanMarkov = 100            # Markov链长度，也即内循环运行次数
    scale   = 0.5               # 定义搜索步长，可以设为固定值或逐渐缩小
    return cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale


# 模拟退火算法
def OptimizationSSA(nVar,xMin,xMax,tInitial,tFinal,alfa,meanMarkov,scale):
    # ====== 初始化随机数发生器 ======
    randseed = random.randint(1, 100)
    random.seed(randseed)  # 随机数发生器设置种子，也可以设为指定整数

    # ====== 随机产生优化问题的初始解 ======
    xInitial = np.zeros((nVar))   # 初始化，创建数组
    for v in range(nVar):
        # xInitial[v] = random.uniform(xMin[v], xMax[v]) # 产生 [xMin, xMax] 范围的随机实数
        xInitial[v] = random.randint(xMin[v], xMax[v]) # 产生 [xMin, xMax] 范围的随机整数
    # 调用子函数 cal_Energy 计算当前解的目标函数值
    fxInitial = cal_Energy(xInitial, nVar, 1) # m(k)：惩罚因子，初值为 1

    # ====== 模拟退火算法初始化 ======
    xNew = np.zeros((nVar))         # 初始化，创建数组
    xNow = np.zeros((nVar))         # 初始化，创建数组
    xBest = np.zeros((nVar))        # 初始化，创建数组
    xNow[:]  = xInitial[:]          # 初始化当前解，将初始解置为当前解
    xBest[:] = xInitial[:]          # 初始化最优解，将当前解置为最优解
    fxNow  = fxInitial              # 将初始解的目标函数置为当前值
    fxBest = fxInitial              # 将当前解的目标函数置为最优值
    print('x_Initial:{:.6f},{:.6f},\tf(x_Initial):{:.6f}'.format(xInitial[0], xInitial[1], fxInitial))

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
            xNew[v] = round(xNow[v] + scale * (xMax[v]-xMin[v]) * random.normalvariate(0, 1))
            # 满足决策变量为整数，采用最简单的方案：产生的新解按照四舍五入取整
            xNew[v] = max(min(xNew[v], xMax[v]), xMin[v])  # 保证新解在 [min,max] 范围内

            # ---计算目标函数和能量差
            # 调用子函数 cal_Energy 计算新解的目标函数值
            fxNew = cal_Energy(xNew, nVar, kIter)
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

        if kIter%10 == 0:                           # 模运算，商的余数
            print('i:{},t(i):{:.2f}, badAccept:{:.6f}, f(x)_best:{:.6f}'.\
                format(kIter, tNow, pBadAccept, fxBest))

        # 缓慢降温至新的温度，降温曲线：T(k)=alfa*T(k-1)
        tNow = tNow * alfa
        kIter = kIter + 1
        fxBest = cal_Energy(xBest, nVar, kIter)  # 由于迭代后惩罚因子增大，需随之重构增广目标函数
        # ====== 结束模拟退火过程 ======

    print('improve:{:d}'.format(totalImprove))
    return kIter,xBest,fxBest,fxNow,recordIter,recordFxNow,recordFxBest,recordPBad

# 结果校验与输出
def ResultOutput(cName,nVar,xBest,fxBest,kIter,recordFxNow,recordFxBest,recordPBad,recordIter):
    # ====== 优化结果校验与输出 ======
    fxCheck = cal_Energy(xBest, nVar, kIter)
    if abs(fxBest - fxCheck)>1e-3:   # 检验目标函数
        print("Error 2: Wrong total millage!")
        return
    else:
        print("\nOptimization by simulated annealing algorithm:")
        for i in range(nVar):
            print('\tx[{}] = {:.1f}'.format(i,xBest[i]))
        print('\n\tf(x) = {:.1f}'.format(cal_Energy(xBest,nVar,0)))

    return


# 主程序
def main():

    # # 参数设置，优化问题参数定义，模拟退火算法参数设置
    # [cName, nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale] = ParameterSetting()
    # # print([nVar, xMin, xMax, tInitial, tFinal, alfa, meanMarkov, scale])

    # # 模拟退火算法    
    # [kIter,xBest,fxBest,fxNow,recordIter,recordFxNow,recordFxBest,recordPBad] = OptimizationSSA(nVar,xMin,xMax,tInitial,tFinal,alfa,meanMarkov,scale)
    # # print(kIter, fxNow, fxBest, pBadAccept)

    # # 结果校验与输出
    # ResultOutput(cName, nVar,xBest,fxBest,kIter,recordFxNow,recordFxBest,recordPBad,recordIter)

    for i in range(1000):
        print(random.normalvariate(0, 1))


if __name__ == '__main__':
    main()