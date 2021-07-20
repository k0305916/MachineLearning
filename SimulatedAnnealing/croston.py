import numpy as np
def Croston(ts, extra_periods=28, alpha=0.1):
    cols = ts.shape[0]
    d = np.append(ts, [np.nan]*extra_periods)
    
    #level (a), periodicity(p) and forecast (f)
    a,p,f = np.full((3,cols+extra_periods),np.nan)
    q = 1
    
    # Initialization
    first_occurence = np.argmax(d[:cols]>0)
    a[0] = d[first_occurence]
    p[0] = 1 + first_occurence
    f[0] = a[0]/p[0]

    for t in range(0, cols):        
        if d[t] > 0:
            a[t+1] = alpha*d[t] + (1-alpha)*a[t] 
            p[t+1] = alpha*q + (1-alpha)*p[t]
            f[t+1] = a[t+1]/p[t+1]
            q = 1           
        else:
            a[t+1] = a[t]
            p[t+1] = p[t]
            f[t+1] = f[t]
            q += 1
       
    # Future Forecast 
    a[cols+1:cols+extra_periods] = a[cols]
    p[cols+1:cols+extra_periods] = p[cols]
    f[cols+1:cols+extra_periods] = f[cols]
                      
    return np.mean((f[:cols] - d[:cols])**2), f


ts = [362.35, 0.0, 0.0, 0.0, 0.0, 361.63, 0.0, 0.0, 0.0, 0.0, 362.77,
      0.0, 0.0, 0.0, 0.0, 356.11, 0.0, 0.0, 0.0, 0.0, 0.0, 331.26,
      0.0, 0.0, 0.0, 0.0, 304.87, 0.0, 0.0, 0.0, 0.0, 0.0, 311.06,
      0.0, 0.0, 0.0, 0.0, 323.62, 0.0, 0.0, 0.0, 0.0, 336.57, 0.0,
      0.0, 0.0, 0.0, 342.40, 0.0, 0.0, 0.0, 0.0, 343.57, 0.0, 0.0,
      0.0, 0.0, 349.56, 0.0, 0.0, 0.0, 0.0, 356.45, 0.0, 0.0, 0.0,
      0.0, 362.56, 0.0, 0.0, 0.0, 0.0, 360.64, 0.0, 0.0, 0.0, 0.0,
      312.74, 0.0, 0.0, 0.0, 0.0]

tss = [0.0, 312.74, 0.0, 0.0, 0.0, 0.0]


dataset = np.concatenate((ts, tss))
result = Croston(dataset, 6, 0.1)
print(("0.1 result:{0}").format(result[0]))
result = Croston(dataset, 6, 0.15)
print(("0.15 result:{0}").format(result[0]))
result = Croston(dataset, 6, 0.2)
print(("0.2 result:{0}").format(result[0]))
result = Croston(dataset, 6, 0.25)
print(("0.25 result:{0}").format(result[0]))
result = Croston(dataset, 6, 0.3)
print(("0.3 result:{0}").format(result[0]))