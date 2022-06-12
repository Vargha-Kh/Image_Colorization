import os
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(home, channels_first=True, train_percent=0.8):
    ab1 = np.load(os.path.join(home, "ab/ab", "ab1.npy"))
    ab2 = np.load(os.path.join(home, "ab/ab", "ab2.npy"))
    ab3 = np.load(os.path.join(home, "ab/ab", "ab3.npy"))
    ab = np.concatenate([ab1, ab2, ab3], axis=0)
    #     ab = np.transpose(ab, [0, 3, 1, 2])
    l = np.load(os.path.join(home, "l/gray_scale.npy"))
    return train_test_split(ab, l, train_size=train_percent)
