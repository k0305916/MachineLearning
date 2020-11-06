import numpy as np
import pandas as pd
import random
from croston import croston
import matplotlib.pyplot as plt


# a = np.zeros(50)
# val = np.array(random.sample(range(100,200), 10))
# idxs = random.sample(range(50), 10)

# a = np.zeros(7)
# val = [1.0,4.0,5.0,3.0]
# idxs = [1,2-1,6-2,7-3]

# ts = np.insert(a, idxs, val)

input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
input_data = input_data.fillna(0)
ts = input_data['Feature']

fit_pred = croston.fit_croston(ts, 10, 'original') # croston's method

# fit_pred = croston.fit_croston(ts, 4, 'sba') # Syntetos-Boylan approximation
# fit_pred = croston.fit_croston(ts, 10, 'sbj') # Shale-Boylan-Johnston


yhat = np.concatenate([fit_pred['croston_fittedvalues'], fit_pred['croston_forecast']])

opt_model = fit_pred['croston_model']
print("opt P: {0}   Q: {1}".format(opt_model["a_demand"],opt_model["a_interval"]))

# print(ts)
# print(yhat)

plt.plot(ts)
plt.plot(yhat)

plt.show()