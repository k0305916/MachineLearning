import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import main

input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
input_data = input_data.fillna(0)
ts = input_data['Feature']

# print(input_data)

# plt.plot(ts['Year'], ts['Feature'])
# plt.show()



# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]

# ts = np.insert(a, idxs, val)

min_rmse = 99999999
min_p = 0
min_q = 0

max_p = len(ts)

# 如何最优化p,q的过程~  重点~
for p in range(1,max_p):
    for q in range(0,p+1):
        fit_pred = main.fit_ata(ts, 4, p, q) # ata's method

        frc_in = fit_pred['ata_fittedvalues']

        rmse = main._ata_cost(ts, frc_in)

        if(rmse < min_rmse):
            min_rmse = rmse
            min_p = p
            min_q = q
        print(("rms: {0} p: {1} q: {2}").format(rmse,p,q))

print(("min_p: {0}  min_q: {1}").format(min_p,min_q))

fit_pred = main.fit_ata(ts, 4, min_p, min_q) # ata's method
yhat = np.concatenate([fit_pred['ata_fittedvalues'], fit_pred['ata_forecast']])

print(ts)
print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()