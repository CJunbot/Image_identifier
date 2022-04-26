from utils.CNN import DeepConvNet
from dataset.mnist import load_mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")
y = network.predict(x_test[0:1])
print(np.argmax(y))
print(y_test[0:1])
