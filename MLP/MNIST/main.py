# -*- coding: utf-8 -*- 
from __future__ import print_function, division
import pickle
import gzip
import numpy as np
import random

from numpy.core.fromnumeric import size

def load_data():
    file = gzip.open('MLP/MNIST/mnist.pkl.gz','rb')
    u = pickle._Unpickler(file)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    file.close()
    return training_data, validation_data, test_data

def vectorized_label(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def data_wrapper():
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x,(784,1)) for x in tr_d[0]]
    training_labels = [vectorized_label(x) for x in tr_d[1]]
    training_data = zip(training_inputs, training_labels)

    validation_inputs = [np.reshape(x,(784,1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x,(784,1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return training_data, validation_data, test_data

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# sigmoid函数的导数
def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))

class Network(object):
    def __init__(self, sizes):
        self.numOfLayers = len(sizes)
        self.sizes = sizes

        [print("{0}".format(i)) for i in sizes[1:]]

        # random initial offset & weight
        self.biases = [np.random.randn(i,1) for i in sizes[1:]]
        self.weights = [np.random.randn(j,i) for i, j in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        if test_data:
            len_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            print ("Epoch {0}: ".format(j))
            random.shuffle(training_data)

            # mini_batches是列表中放切割之后的列表
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n, mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nable_w = [np.zeros(w.shape) for w in self.weights]

                eta = learning_rate / len(mini_batch)

                for x,y in mini_batch:
                    # 从一个实例得到的梯度
                    delta_nabla_b, delta_nabla_w = self.backprop(x,y)
                    nabla_b = [nb+dnd for nb,dnd in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.biases = [b-eta*nb for b, nb in zip(self.biases, nabla_b)]
                self.weights = [w-eta*nw for w, nw in zip(self.weights, nabla_w)]

        if test_data:
            print("{0}/{1}={2}".format(self.evaluate(test_data), len_test, self.evaluate(test_data)/len_test))

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向过程
        activation = x
        activations = [x]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # 反向过程
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    def evaluate(self, test_data):
        test_result = [(np.argmax(self.feedforward(x)), y) for (x,y) in test_data]
        return sum(int(i==j) for (i,j) in test_result)


def main():
    # 加载数据集
	train_data, validation_data, test_data = data_wrapper()

	# 训练神经网络
	net_trained = Network([784, 30, 10])
	net_trained.SGD(train_data, 30, 10, 3, test_data)

if __name__ == '__main__':
    main()