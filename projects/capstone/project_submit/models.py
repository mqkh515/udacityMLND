import numpy as np
import data_prep


def median_model(train_x, train_y, test_x):
    return np.ones(test_x.shape[0]) * train_y.median()
