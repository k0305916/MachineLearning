
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Practice\\1\\Data\\FitData_1504.txt",sep="\t",header=None)
data.columns=['Centroid','Distance']
newdata = data.groupby(['Distance']).mean()



plt.scatter(data['Distance'],data['Centroid'],marker='o',s=3)
plt.scatter(newdata.index,newdata['Centroid'],marker='s',color='red',s=3)
plt.ylabel('Centroid')
plt.xlabel('Distance')
plt.show()
