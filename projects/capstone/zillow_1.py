import numpy as np
import pandas as pd
import gc
from time import time
import matplotlib.pyplot as plt

# init load data
prop_data = pd.read_csv('data/properties_2016.csv', header=0)
error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
error_data['sale_month'] = error_data['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month

train_data = error_data.merge(prop_data, how='left', on='parcelid')
train_data.to_csv('data/train_data_merge.csv', index=False)


def multi_trade_analysis(error_data_inp):
    """an asset can be traded more than once within a year, can multi_trade related to large error? should we exclude these samples from training?"""
    error_data = error_data_inp.copy()
    print('average abs logerror in all traning transactions: %4.4f' % np.abs(error_data['logerror']).mean())  # 0.0684
    n_trade = error_data['logerror'].groupby(error_data['parcelid']).count()
    multi_trade_count = n_trade[n_trade > 1]
    print('number of parcels traded more than once in training data: %d' % len(multi_trade_count))  # 124, number too small
    multi_trade_parcel_set = set(multi_trade_count.index.values)
    error_data['is_multi_trade'] = error_data['parcelid'].apply(lambda x: x in multi_trade_parcel_set)
    error_data_multi_trade = error_data[error_data['is_multi_trade']]
    print('average abs logerror in all multi_trade transactions: %4.4f' % np.abs(error_data_multi_trade['logerror']).mean())  # 0.0921


def property_cleaning():
    pass


def cat_property_analysis():
    pass


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

