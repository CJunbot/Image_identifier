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



batch_size = 100
accuracy = 0

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

with open("./dataset/sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

for i in range(0, len(x_train), batch_size):
    x_batch = x_train[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # 각 행에서 최댓값(100,10)이니 100행에서 각각 하나씩 -> (100,1)
    accuracy += np.sum(p == y_train[i:i+batch_size])


print("accuracy: ", accuracy/len(x_train))
