# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from utils.Two_layer import TwoLayerNet
from utils.Optimizer import Adam


optimize = Adam
(x_train, y_train), (x_test, y_test) = load_mnist(one_hot_label=True, normalize=True)

repeat_number = 10000
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(784, 100, 10)
train_acc_list = []
test_acc_list = []
loss_list = []
train_size = x_train.shape[0]

iter_per_epoch = max(train_size / batch_size, 1)
# epoch(에폭) = 훈련데이터 크기 / 배치크기

for i in range(repeat_number):
    batch_index = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_index]
    y_batch = y_train[batch_index]

    grad = network.gradient(x_batch, y_batch)
    optimize.update(network.params, grad)
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]  #SGD


    loss = network.loss(x_batch, y_batch)
    loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_acc_list.append(test_acc)
        print("train acc:" + str(train_acc) + "| test acc:" + str(test_acc))

