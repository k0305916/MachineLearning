# 为了使用Python代码实现加权多数投票，可以使用NumPy中的argmax和bincount函数：

# 为实现基于类别预测概率的加权多数投票，我们可以再次使用NumPy中的numPy.average和np.argmax方法：

# 综合上述内容，我们使用Python来实现MajorityVoteClassifier类：


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
# 此外请注意，我们导入six包从而使得MajorityVoteClassifier与Python 2.7兼容。
from sklearn.externals import six 
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator


# 主要是出于演示的目的，我们才实现了MajorityVoteClassifier类，
# 不过这也完成了scikit-learn中关于多数投票分类器的一个相对复杂的版本。
# 它将出现在下一个版本（v0.17）的sklearn.ensemble.VotingClassifier类中。
class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    """ A majority vote ensemble classifier

    Parameters
    ------------
    classifiers: array-like, shape=[n_classifiers]
        Different classifiers for the ensemble
     
     vote: str,{'classlabel','probability'}
        Default: 'classlabel'
        If 'classlabel' the prediction is based on
        the argmax of class labels. Else if 'probability',
        the argmax of the sum of probabilities is used to predict 
        the class label(recommended for calibrated classifiers).


    weights: array-like,shape=[n_classifiers]
        Optional, default: None
        If a list of 'int' or 'float' values are
        provided, the classifiers are weighted by 
        importance; Uses uniform weights if 'weights=None'.

    """
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key:value for key,value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self,x,y):
        """Fit classifiers.

        Parameters
        ----------
        x: {array-like,sparse matrix},
            shape=[n_samples,n_features]
            Matrix of training samples.

        y: {array-like},shape=[n_samples]
            Vector of target class labels.

        Returns
        -------
        self: object
        """
        # Use LabelEncoder to ensure class labels start
        # with 0, which is important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_=self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(x,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self


    def predict(self,x):
        """ Predict class labels for x.

        Parameters
        ----------
        x: {array-like, sparse matrix},
            Shape=[n_samples,n_features]
            Matrix of training samples.
        
        Returns
        --------
        maj_vote: array-like, shape=[n_samples]
            Predicted class labels.

        """
        if self.vote='probability':
            maj_vote = np.argmax(self.predict_proba(x),axis=1)
        else: #'classlabel' vote
            # Collect results from clf.predict calls
            predictions = np.asarray([clf.predict(x) for clf in self.classifiers_]).target
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x,weights=self.weights)),axis=1,arr=predictions)
            maj_vote = self.lablenc_.inverse_transform(maj_vote)
        
        return maj_vote

    def predict_proba(self,x):
        """ Predict class probabilities for x.


        Parameters
        ----------
        x: {array-like, sparse matrix},
            Shape=[n_samples,n_features]
            Training vectors, where n_samples is 
            the number of samples and n_feature is the 
            number of features.

        Returns
        ---------
        avg_proba: array-like,
            Shape=[n_samples,n_classes]
            Weighted average probability for
            each class per sample.

        """
        probas = np.asarray([clf.predict_proba(x) for clf in self.classifiers])
        avg_proba = np.average(probas,axis=0,weights=self.weights)

        return avg_proba

    def get_params(self,deep=True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name,step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s'%(name,key)]=value
            return out


from sklean import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
iris = datasets.load_iris()
x,y = iris.data[50:,[1,2]],iris.target[50:]
le = LabelEncoder()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=1)

# 我们使用训练数据集训练三种不同类型的分类器：逻辑斯谛回归分类器、决策树分类器及k-近邻分类器各一个，在将它们组合成集成分类器之前，我们先通过10折交叉验证看一下各个分类器在训练数据集上的性能表现：
from sklearn.cross_validation import corss_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# 读者可能会感到奇怪，为什么我们将逻辑斯谛回归和k-近邻分类器的训练作为流水线的一部分？
# 原因在于：如我们在第3章中所述，不同于决策树，逻辑斯谛回归与k-近邻算法（使用欧几里得距离作为距离度量标准）对数据缩放不敏感。
# 虽然鸢尾花特征都以相同的尺度（厘米）度量，不过对特征做标准化处理是一个好习惯。
clf1 = LogisticRegression(penalty='l2',C=0.001,random_state=0)
clf2 = DecisionTreeClassifier(max_depth=1,criterion='entropy',random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')
pipe1 = Pipeline([['sc',StandardScaler()],['clf',clf1]])
pipe3= Pipeline([['sc',StandardScaler()],['clf',clf3]])
clf_labels = ['Logistic Regression','Decision Tree','KNN']
for clf,label in zip([pipe1,clf2,pipe3],clf_labels):
    scores=corss_val_score(estimator=clf,X=x_train,y=y_train,cv=10,scoring='roc_auc')
    print('ROC AUC: %0.2f (+/- %0.2f) [%s]' %(scores.mean(),scores.std(),label))



mv_clf = MajorityVoteClassifier(classifiers=[pipe1,clf2,pipe3])
clf_labels+=['Majority Voting']
all_clf = [pipe1,clf2,pipe3,mv_clf]
for clf,label in zip(all_clf,clf_labels):
    #正如我们所见，以10折交叉验证作为评估标准，MajorityVotingClassifier的性能与单个成员分类器相比有着质的提高。
    scores = corss_val_score(estimator=clf,X=x_train,y=y_train,cv=10,scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(),scores.std(),label))

