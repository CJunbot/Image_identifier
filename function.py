import numpy as np


def identity_function(x):
    return x


def relu(x):
    return np.maximum(0,x)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(y):
    c=np.max(y)
    exp=np.exp(y-c)
    sum_exp=np.sum(exp)
    z=exp/sum_exp
    return z
# 출력의 총합이 1(확률로 해석), 출력층 함수, 분류에 사용



