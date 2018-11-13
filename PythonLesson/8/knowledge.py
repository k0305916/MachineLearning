import pyprind
import pandas as pd
import os

#Knowledge---------------------
# bag-of-word model
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# 借助于scikit-learn中的CountVecorizer类，
# 我们可以通过设置其ngram_range参数来使用不同的n元组模型。
# 此类默认为1元组，使用ngram_range=（2,2）初始化一个新的CountVectorizer类，可以得到一个2元组表示。


# 例如，Kanaris等人通过研究发现，在反垃圾邮件过滤中，n的值为3或者4的n元组即可得到很好的效果.
# （Ioannis Kanaris,Konstantinos Kanaris,Ioannis Houvardas,and Efstathios 
# Stamatatos.Words vs Character N-Grams for Anti-Spam Filtering.International Journal 
# on Artificial Intelligence Tools,16（06）:1047–1067,2007）
count= CountVectorizer()
# count2 = CountVectorizer(ngram_range=(2,2))
docs = np.array(['The sun is shining',
    'The weather is sweet',
    'The Sun is shining and the weather is sweet'])
# bag = count.fit_transform(docs)
# bag2 = count2.fit_transform(docs)
# print(count.vocabulary_)  #下面矩阵的抬头序号
# print(bag.toarray())

# print(count2.vocabulary_)
# print(bag2.toarray())


# tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())


# clean unuseful words---important step
import re
def preprocessor(text):
    text = re.sub('<[^>*>','',text)
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+',' ',text.lower()).join(emotions).replace('-','')
    return text

preprocessor(df.loc[0,'review'][-50:])
preprocessor("</a>This :) is :( a test :-)!")
df['review'] = df['review'].apply(preprocessor)

# remark documents
# a way: split words through space(' ').
def tokenizer(text):
    return text.split()

#test
tokenizer('runners like running and thus they run')

# other way: word stemming
# 最初的词干提取算法是由Martin F.Porter于1979年提出的，由此也称为Porter Stemmer算法[1]。
# Python自然语言工具包（NLTK,http://www.nltk.org, http://www.nltk.org/book/）实现了Porter Stemming算法，
# 我们将在下一小节中用到它。要安装NLTK，只要执行命令：pip install nltk。
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#test  不过，相较于词干提取，词形还原的计算更加复杂和昂贵
tokenizer_porter('runners like running and thus they run')


#停用词移除（stop-word removal）
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]
#Knowledge---------------------