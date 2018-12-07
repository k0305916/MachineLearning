

# Serialization
import pickle
import os
dest = od.path.join('moiveclassifier','pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

# 使用pickle中的dump方法，对训练好的逻辑斯谛回归模型及NLTK库中的停用词进行序列化，这样就不必在Web应用服务器上安装NLTK库了。
pickle.dump(stop,open(os.path.join(dest,'stopwords.pkl'),'wb'),
        protocol=4)
pickle.dump(clf,open(os.path.join(dest,'classifier.pkl'),'wb'),
        protocol=4)

import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(os.path.join('pkl_objects','classifier.pkl'),'rb'))


# 已经经过了反序列化后，就可以使用了
import numpy
label = {0:'negative',1:'positive'}
example = ['I love this movie']
X = vect.transform(example)
# 请注意，对predict_proba方法的调用会返回一个数组，以及每个类标所对应的概率。由于predict返回的是较高概率对应的类标，我们则使用np.max函数返回对应预测类别的概率[3]。
print('Prediction: %s\nProbability: %.2f%%' %(label[clf.predict(X)[0]],np.max(clf.predict_proba(X))*100))



