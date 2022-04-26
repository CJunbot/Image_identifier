import pickle
from utils.function import sigmoid, softmax
import numpy as np
from dataset.mnist import load_mnist


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


accuracy = 0

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

with open("./dataset/sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

batch_size = 100
batch_index = np.random.choice(x_train.shape[0], batch_size)
x_batch = x_train[batch_index]
y = predict(network, x_batch)
compare_y = np.argmax(y, axis=1)
y_batch = y_train[batch_index]


accuracy += np.sum(compare_y == y_batch)

print("accuracy: ", accuracy/batch_size)
