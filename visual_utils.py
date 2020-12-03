import os
import cv2
import math
import torch
import numpy as np

def visual_weights(weights, save_dir, name, channels=3):
    n = weights.shape[0]
    for idx in range(n):
        image_size = int(math.sqrt(int(weights.shape[1]) / channels))
        image = weights[idx].reshape(channels, image_size, -1)
        image = image.transpose((2, 1, 0))

        cv2.imwrite(os.path.join(save_dir, "{}_{}.jpg".format(name, idx)), np.array(image, np.uint8))


