import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.DataFrame([
    ['green', 'M', '10.1', 'class1'],
    ['red', 'L', '13.5', 'class2'],
    ['blue', 'XL', '15.3', 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']

# 如果在后续过程中需要将整数值还原为有序字符串，可以简单地定义一个逆映射字典：
# inv_size_mapping={v:k for k,v in size_mappping.items()}，
# 与前面用到的size_mapping类似，可以通过pandas的map方法将inv_size_mapping应用于经过转换的特征列上。
# size_mapping ={'XL':3,'L':2,'M':1}
# df['size'] = df['size'].map(size_mapping)

# class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
# df['classlabel'] = df['classlabel'].map(class_mapping)

# inv_class_mapping={v:k for k,v in class_mapping.items()}
# df['classlabel'] = df['classlabel'].map(inv_class_mapping)

# more easy way to translate class label
# class_le = LabelEncoder()
# y = class_le.fit_transform(df['classlabel'].values)

# class_le.inverse_transform(y)

# 将颜色按照int类型来处理，会产生错误：颜色之间是没有大小之分的。
# 解决此问题最常用的方法就是独热编码（one-hot encoding）技术。
# x = df[['color', 'size', 'price']].values
# color_le = LabelEncoder()
# x[:, 0] = color_le.fit_transform(x[:, 0])

# 当我们调用OneHotEncoder的transform方法时，它会返回一个稀疏矩阵。
# 出于可视化的考虑，我们可以通过toarray方法将其转换为一个常规的NumPy数组。
# 稀疏矩阵是存储大型数据集的一个有效方法，被许多scikit-learn函数所支持，
# 特别是在数据包含很多零值时非常有用。
# ohe = OneHotEncoder(categorical_features=[0])
# ohe.fit_transform(x).toarray()

# 另外，我们可以通过pandas中的get_dummies方法，更加方便地实现独热编码技术中的虚拟特征。
# 当应用于DataFrame数据时，get_dummies方法只对字符串列进行转换，而其他的列保持不变。
pd.get_dummies(df[['price','color','size']])

print(df)
