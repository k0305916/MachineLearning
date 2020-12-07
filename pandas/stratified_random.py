import pandas as pd

df = pd.DataFrame(dict(
           A=[1, 1, 1, 2, 2, 2, 2, 3, 4, 4],
           B=range(10)
       ))

print(df)

# df.groupby('A', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))
result = df.groupby('A', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))

print(result)
