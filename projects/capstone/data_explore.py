import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_train_data():
    prop_data = pd.read_csv('data/properties_2016.csv', header=0)
    error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
    train_data = error_data.merge(prop_data, on='parcelid')