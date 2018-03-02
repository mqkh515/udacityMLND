from catboost import CatBoostRegressor
import catboost
import numpy as np
import pandas as pd



def cat_boost_model_prep(data):
    """mark categorical column indexes"""
    pass


def cat_boost_test():
    x = pd.DataFrame(np.random.randn(1000, 3))
    y = x[0] + 1 + x[1] + 2 + x[2] * 3 + np.random.randn(1000) / 2
    res = catboost.cv({}, catboost.Pool(x, y), 5)