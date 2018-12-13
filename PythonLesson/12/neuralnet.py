import os
import struct
import numpy as np

def load_mnist(path,kind='train'):
    """Load MNIST data from 'path'"""
    labels_path = os.path.join(path,'%s-balels-idx1-ubyte',%kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte',%kind)
    with open(labels_path,'rb') as lbpath:
        magic,n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.unit8)
    
    with open(images_path,'rb') as imgpath:
        magic,num,rows,cols = struct.unpack('>III',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.unit8).reshape(len(labels),784)

    return images,labels


x_train,y_train = load_mnist('mnist',kind='train')
print('Rows: %d, columns: %d' %(x_train.shape[0],x_train.shape[1]))
x_test,y_test = load_mnist('mnist',kind='t10k')
print('Rows: %d, columns: %d' % (x_test.shape[0],x_test.shape[1]))

#  为了解MNIST数据集中图像的样子，我们通过将特征矩阵中的784像素向量还原为28×28图像，并使用matplotlib中的imshow函数将0～9数字的示例进行可视化展示：
import matplotlib.pyplot as plt
fig,ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(10):
    img = x_train[y_train == i][0].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 此外，我们再绘制一下相同数字的多个示例，来看一下这些手写样本之间到底有多大差异：
fig,ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = x_train[y_train==7][i].reshape(28,28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

# 在将MNIST数据加载到NumPy数组中后，我们可以在Python中执行如下代码，即可将数据存储为CSV格式文件：
np.savetext('Train_img.csv',x_train,fmt='%i',delimiter=',')
np.savetext('Train_labels.csv',y_train,fmt='%i',delimiter=',')
np.savetext('Test_img.csv',x_test,fmt='%i',delimiter=',')
np.savetext('Test_labels.csv',y_test,fmt='%i',delimiter=',')

# 对于已经保存过的CSV格式文件，我们可以使用NumPy的genfromtxt函数对其进行加载：
x_train = np.genfromtxt('Train_img.csv',dtype=int,delimiter=',')
x_train = np.genfromtxt('Train_labels.csv',dtype=int,delimiter=',')
x_train = np.genfromtxt('Test_img.csv',dtype=int,delimiter=',')
x_train = np.genfromtxt('Test_labels.csv',dtype=int,delimiter=',')

import numpy as np
from scipy.special import expit
import sys

class NeuralNetMLP(object):
    def __init__(self,n_output,n_features,n_hidden=30,l1=0.0,l2=0.0,epochs=500,
    eta=0.001,alpha =0.0,decrease_const=0.0,shuffle=True,minibatches=1,random_state=None):
    np.random.seed(random_state):
        self.n_output = n_output
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.w1,self.w2 = self._initialize_weights()
        self.l1 = l1
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.alpha = alpha
        self.decrease_const = decrease_const
        self.shuffle = shuffle
        self.minibatches = minibatches

    def _encode_labels(self,y,k):
        onehot = np.zeros((k,y.shape[0]))
        for idx,val in enumerate(y):
            onehot[val,idx] = 1.0
        return onehot

    def __initialize_weights(self):
        w1 = np.random.uniform(-1.0,1.0,size=self.n_hidden*()self.n_features+1)
        w1 = s1.reshape(self.n_hidden,self.n_features+1)
        w2 = np.random.uniform(-1.0,1.0,size=self.n_output*(self.n_hidden+1))
        w2 = w2.reshape(self.n_output,self.n_hidden+1)
        return w1,w2

    def _sigmoid(self,z):
        # expit is equivalent to 1.0/(1.0 + np.exp(-z))
        return expit(z)

    def _sigmoid_gradient(self,z):
        sg = self._sigmoid(z)
        return sg*(1-sg)

    def _add_bias_unit(self,x,how='column'):
        if how=='column':
            x_new = np.ones((x.shape[0],x.shape[1]+1))
            x_new[:,1:]=x
        elif how=='row':
            x_new = np.ones((x.shape[0]+1,x.shape[1]))
            x_new[1:,:] =x
        else:
            raise AttributeError("How must be 'column' or 'row'")
        return x_new

    def _feedforward(self,x,w1,w2):
        a1 = self._add_bias_unit(x,how='column')
        z2 = w1.dot(a1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2,how='row')
        z3=w2.dot(a2)
        a3 = self._sigmoid(z3)
        return a1,z2,a2,z3,a3

    def _L2_reg(self,lambda_,w1,w2):
        return (lambda_/2.0)*(np.sum(w1[:,1:]**2)+np.sum(w2[:,1:]**2))

    def _L1_reg(self,lambda_,w1,w2):
        return (lambda/2.0)*(np.abs(w1[:,1:]).sum()+np.abs(w2[:,1:]).sum())

    def _get_cost(self,y_enc,output,w1,w2):
        term1 = -y_enc*(np.log(output))
        term2 = (1-y_enc)*np.log(1-output)
        cost = np.sum(term1-term2)
        l1_term = self._L1_reg(self.l1,w1,w2)
        l2_term = self._L2_reg(self.l2,w1,w2)
        cost = cost+l1_term+l2_term
        return cost

    def _get_gradient(self,a1,a2,a3,z3,y_enc,w1,w2):
        #backpropagation
        sigma3 = a3-y_enc
        z2 = self._add_bias_unit(z2,how='row')
        sigma2 = w2.t.dot(sigma3)*self._sigmoid_gradient(z2)
        sigma2 = sigma2[1:,:]
        grad1 = sigma2.dot(a1)
        grad2 = simga3.dot(a2.T)

        #regularize
        grad1[:,1:] += (w1[:,1:]*(self.l1+self.l2))
        grad2[:,1:] += (w2[:,1:]*(self.l1+self.l2))

        return grad1,grad2

    def predict(self,x):
        a1,z2,a3,z3,a3 = self._feedforward(x,self.w1,self.w2)
        y_pred = np.argmax(z3,axis=0)
        return y_pred

    def fit(self,x,y,print_pregress=False):
        self.cost_ = []
        x_data,y_data = x.copy(),y.copy()
        y_enc = self._encode_labels(y,self.n_output)

        delta_w1_prev = np.zeros(self.w1.shape)
        delta_w2_prev = np.zeros(self.w2.shape)

        for i in range(self.epochs):
            #adaptive learning rate
            self.eta /= (1+self.decrease_const*i)

            if print_pregress:
                sys.stderr.write('\rEpoch: %d/%d' % (i+1,self.epochs))
                sys.stderr.flush()

            if self.shuffle:
                idx = np.random.pernutation(y_data.shape[0])
                x_data,y_data = x_data[idx],y_data[idx]

            mini = np.array.split(range(y_data.shape[0]),self.minibatches)

            for idx in mini:
                #feedforward
                a1,z2,a2,z3,a3 = self._feedforward(x[idx],self.w1,self.w2)
                cost = self._get_cost(y_enc=y_enc[:,idx],outpu=a3,w1=self.w1,w2=self.w2)
                self.cost_.append(cost)

            #compute gradient via backpropagation
            grad1,grad2 = self._get_gradient(a1=a1,a2=a2,a3=a3,z2=z2,y_enc=y_enc[:,idx],w1=self.w1,w2=self.w2)

            #update weights
            delta_w1,delta_w2 = self.eta*grad1,self.eta*grad2
            self.w1 -=(delta_w1+(self.alpha*delta_w1_prev))
            self.w2 -=(delta_w2+(self.alpha*delta_w2_prev))
            delta_w1_prev,delta_w2_prev = delta_w1,delta_w2
        return self


# 我们来初始化一个784-50-10的感知器模型，该神经网络包含784个输入单元（n_features），50个隐层单元（n_hidden），以及10个输出单元（n_output）：
# ·l2：L2正则化系数λ，用于降低过拟合程度，类似地，l1对应L1正则化参数λ。
# ·epochs：遍历训练集的次数（迭代次数）。
# ·eta：学习速率η。
# ·alpha：动量学习进度的参数，它在上一轮迭代的基础上增加一个因子，用于加快权重更新的学习Δwt=η▽J（wt+Δwt-1）（其中，t为当前所在的步骤，也就是当前迭代次数）。
# ·decrease_const：用于降低自适应学习速率n的常数d，随着迭代次数的增加而随之递减以更好地确保收敛。
# ·shuffle：在每次迭代前打乱训练集的顺序，以防止算法陷入死循环。
# ·Minibatches：在每次迭代中，将训练数据划分为k个小的批次，为加速学习的过程，梯度由每个批次分别计算，而不是在整个训练数据集上进行计算。
nn = NeuralNetMLP(n_output=10,
                n_features=x_train.shape[1],
                n_hidden=50,
                l2=0.1,
                l1 = 0.0,
                epochs = 1000,
                eta = 0.001,
                alpha = 0.001,
                decrease_cost=0.00001,
                shuffle=True,
                minibatches=50,
                reandom_state=1)

nn.fit(x_train,y_train,print_progress=True)

# 我们使用cost_列表保存了每轮迭代中的代价并且可以进行可视化，以确保优化算法能够收敛。在此，我们仅绘制了50个子批次的每50次迭代的结果（50个子批次×1000次迭代）。
plt.plot(range(len(nn.cost_)),nn.cost_)
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

# 绘制出了一个相对平滑的代价函数图像
batches = np.array_split(range(len(nn.cost_)),1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]

plt.plot(range(len(cost_avgs)),cost_avgs,color='red')
plt.ylim([0,2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()

# 我们通过计算预测精度来评估模型的性能：
y_train_pred = nn.predict(x_train)
acc = np.sum(y_train==y_train_pred,axis=0) / x_train.shape[0]
print('Training accuracy: %.2f%%' %(acc*100))

# 正如我们所见，模型能够正确识别大部分的训练数字，不过现在还不知道将其泛化到未知数据上的效果如何？我们来计算一下模型在测试数据集上10000个图像上的准确率：
y_test_pred = nn.predict(x_test)
acc = np.sum(y_test==y_test_pred,axis=0)/x_test.shape[0]
print('Testing accuracy: %.2f%%' % (acc*100))


# 我们看一下多层感知器难以处理的一些图像：
miscl_img = x_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test!=y_test_pred][:25]
miscl_lab = y_test_pred[y_test!=y_test_pred][:25]
fig,ax = plt.subplots(nrows = 5,ncols = 5,sharex=true,sharey=true)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28,28)
    ax[i].imshow(img,cmp='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1,correct_lab[i],miscl_lab[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
# 第一个数字为图像索引，第二个数字为真实的类标（t），第三个数字则是预测的类标（p）
plt.show()









