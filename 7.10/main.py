import pandas as pd

dataset = pd.read_csv('data/DataSet.csv')

#do not care about the score. Development first.
score = {'dark_green':1,'black':2,'light_white':3,
        'curl_up':1,'little_curl_up':2,'stiff':3,
        'little_heavily':1,'heavily':2,'clear':3,
        'distinct':1,'little_blur':2,'blur':3,
        'sinking':1,'little_sinking':2,'even':3,
        '1':1,'0':0}

