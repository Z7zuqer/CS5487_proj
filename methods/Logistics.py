import os
import numpy as np

class Logistics(object):
    def __init__(self, input_num, class_num, lr, channels=3):
        self.class_num = class_num
        self.input_num = input_num * channels
        self.weights = np.ones((input_num * channels, class_num))
        self.lr = lr

    def __call__(self, inputx, label):
        inputx = np.mat(inputx.reshape(-1, self.input_num).numpy())
        label = self._convert_one_hot(label)

        x = 1.0 / (1 + np.exp(-1 * inputx * self.weights))

        loss = x - label

        grad = np.matmul(inputx.transpose(), loss)

        self.weights -= grad * self.lr

        return abs(loss).mean()

    def get_accuracy(self, inputx, label):
        inputx = np.mat(inputx.reshape(-1, self.input_num).numpy())

        x = 1.0 / (1 + np.exp(-1 * (inputx * self.weights)))

        pred = self._convert_from_one_hot(x)

        acc = (pred==label.numpy()).mean()
        return acc

    def _convert_one_hot(self, label):
        new_label = np.zeros((label.shape[0], self.class_num))
        for idx, val in enumerate(label):
            new_label[idx][val] = 1
        return np.mat(new_label)

    def _convert_from_one_hot(self, outputs):
        res = np.ones(outputs.shape[0])
        outputs = outputs.tolist()
        for idx, item in enumerate(outputs):
            max_value, max_idx = -10000., -1
            for i_idx, i_item in enumerate(item):
                if max_value < i_item:
                    max_value = i_item
                    max_idx = i_idx
            res[idx] = max_idx
        return res


