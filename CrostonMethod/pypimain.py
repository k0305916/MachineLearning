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

ts = [362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
      0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
      0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
      0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
      0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
      0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
      0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
      312.74, 0.0, 0.0, 0.0, 0.0]

tss = [0.0, 312.74, 0.0, 0.0, 0.0, 0.0]

# input_data = pd.read_csv("./data/M4DataSet/NewYearly.csv")
# input_data = input_data.fillna(0)
# ts = input_data['Feature']

fit_pred = croston.fit_croston(ts, 6, 'original') # croston's method

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