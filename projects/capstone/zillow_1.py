import numpy as np
import pandas as pd
import gc
from time import time
import matplotlib.pyplot as plt


prop_data = pd.read_csv('data/properties_2016.csv', header=0)
error_data = pd.read_csv('data/train_2016_v2.csv', header=0)


def census_analysis(census_series):
    # remove nan
    census_nullremove = census_series[~census_series.isnull()]
    # transform to str list, remove first 4 digits (fips)
    census_str = [str(int(s))[4:] for s in census_nullremove.values]
    # check census str be of the same len
    len_census_str = [len(s) for s in census_str]
    print('num_len_val: %s (expect to be one)' % len(set(len_census_str)))
    print('n_group: %d' % len(set(census_str)))
    # investigate split position
    for i in range(1, len(census_str[0]) - 1):
        max_split1 = float(10 ** i - 1)
        census_split1 = [c[:i] for c in census_str]
        max_split2 = float(10 ** len(census_str[0]) - 1)
        census_split2 = [c[i:] for c in census_str]
        print('%d; ratio_group1: %4.2f; ratio_group2: %4.2f' % (i, len(set(census_split1)) / max_split1, len(set(census_split2)) / max_split2))



