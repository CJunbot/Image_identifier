# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from utils.Two_layer import TwoLayerNet, TwoLayerNet2
from utils.Optimizer import SGD
from utils.function import *

optimize = SGD()
(x_train, y_train), (x_test, y_test) = load_mnist(one_hot_label=True, normalize=True)

repeat_number = 10000
batch_size = 100
learning_rate = 0.1
network = TwoLayerNet(784, 100, 10)
network2 = TwoLayerNet2(784, 100, 10)
train_acc_list = []
test_acc_list = []
adam_loss_list = []
sgd_loss_list = []
train_size = x_train.shape[0]

iter_per_epoch = max(train_size / batch_size, 1)
# epoch(에폭) = 훈련데이터 크기 / 배치크기

for i in range(repeat_number):
    batch_index = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_index]
    y_batch = y_train[batch_index]

    grad = network.gradient(x_batch, y_batch)
    optimize.update(params=network.params, grads=grad)  # sgd와 adam 차이가 엄청 크다
    loss = network.loss(x_batch, y_batch)
    adam_loss_list.append(loss)

    grad = network2.gradient(x_batch, y_batch)
    optimize.update(params=network2.params, grads=grad)
    loss2 = network2.loss(x_batch, y_batch)
    sgd_loss_list.append(loss2)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, y_train)
        train_acc_list.append(train_acc)
        test_acc = network.accuracy(x_test, y_test)
        test_acc_list.append(test_acc)
        print("train acc:" + str(train_acc) + "| test acc:" + str(test_acc))

x = np.arange(repeat_number)
plt.plot(x, smooth_curve(adam_loss_list), marker='o', markevery=100, label='adam')
plt.plot(x, smooth_curve(sgd_loss_list), marker='x', markevery=100, label='sgd')
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(-0.2, 1)
plt.legend()
plt.show()
