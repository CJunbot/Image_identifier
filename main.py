from function import sigmoid, relu
import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)

