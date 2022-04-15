import pickle
from function import sigmoid, relu, softmax
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1= sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

with open("./dataset/sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

accuracy = 0
for i in range(len(x_train)):
    y = predict(network, x_train[i])
    p = np.argmax(y)
    if p == y_train[i]:
        accuracy += 1
print("accuracy: ", accuracy/len(x_train))
