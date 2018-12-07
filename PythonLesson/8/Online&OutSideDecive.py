# 我们将使用scikit-learn中SGDClassifier的partial_fit函数来读取本地存储设备，
# 并且使用小型子批次（minibatches）文档来训练一个逻辑斯谛回归模型。

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')

import numpy as np
import re
def tokenizer(text):
    text = re.sub('<[^>*>','',text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+',' ',text.lower())+' '.join(emotions).replace('-','')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def stream_docs(path):
    with open(path,'r') as csv:
        next(csv) #skip header
        for line in csv:
            text,label = line[:-3],int(line[-2])
            yield text,label

# next(stream_docs(path='./movie_data.csv'))

def get_minibatch(doc_stream,size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
vect = HashingVectorizer(decode_error='ignore',n_feature=2**21,preprocessor=None,tokenizer=tokenizer)
clf = SGDClassifier(loss='log',random_state=1,n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')


# 现在到了真正有趣的部分。设置好所有的辅助函数后，我们可以通过下述代码使用外存学习：
import pyprind
pbar = pyprind.ProgBar(45)
classes = np.array([0,1])
for _ in range(45):
    x_train,y_train = get_minibatch(doc_stream,size=1000)
    if not x_train:
        break
    x_train = vect.transform(x_train)
    clf.partial_fit(x_train,y_train,classes=classes)
    pbar.update()

x_test,y_test = get_minibatch(doc_stream,size=5000)
x_test = vect.transform(x_test)
print('Accuracy: %.3f' % clf.score(x_test,y_test))
# 我们可以通过剩下的5000个文档来升级模型：
clf = clf.partial_fit(x_test,y_test)
