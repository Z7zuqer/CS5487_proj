import os
import numpy as np

class Logistics(object):
    def __init__(self, class_num, lr):
        self.weights = np.ones((class_num, 1))
        self.lr = lr

    def __call__(self, inputx, label):
        inputx = np.mat(inputx.numpy())
        x = 1.0 / (1 + np.exp(-1 * (inputx * self.weights)))
        grad = inputx.transpose() * (x - label)
        self.weights = self.weights - grad * self.lr

    def get_accuracy(self, x, label):
        output = np.exp(x.dot(self.weights))
        output = np.mat([1 if item >= 0.5 else 0 for item in output])
        acc = (output.T==label).mean()
        return acc



