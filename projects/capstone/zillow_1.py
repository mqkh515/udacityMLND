import numpy as np
import pandas as pd
import math
import gc
from time import time
import matplotlib.pyplot as plt

# init load data
prop_data = pd.read_csv('data/properties_2016.csv', header=0)
error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
error_data['sale_month'] = error_data['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month

train_data = error_data.merge(prop_data, how='left', on='parcelid')
train_data.to_csv('data/train_data_merge.csv', index=False)

RENAMING_MAP = {}  # map from old name to new name, for column rename, getting more meaningful names
KEEP_FEATURE = set()


def cat_num_to_str(col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not prop_data[col_name].dtype == 'O':
        prop_data[col_name] = prop_data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)
    if not train_data[col_name].dtype == 'O':
        train_data[col_name] = train_data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


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


def property_cleaning_naive():
    """for Udacity capstone, minor and naive handling of missing values"""
    global prop_data, train_data

    # type_air_conditioning, missing rate 0.6812
    RENAMING_MAP['airconditioningtypeid'] = 'type_air_conditioning'
    KEEP_FEATURE.add('airconditioningtypeid')
    cat_num_to_str('airconditioningtypeid')
    prop_data['airconditioningtypeid'].loc[prop_data['airconditioningtypeid'].index[prop_data['airconditioningtypeid'] == '12']] = np.nan  # setting 12 in prop data to nan

    # type_architectural_style, missing rate 0.9971, dropped
    RENAMING_MAP['architecturalstyletypeid'] = 'type_architectural_style'
    cat_num_to_str('architecturalstyletypeid')
    prop_data['architecturalstyletypeid'].loc[prop_data['architecturalstyletypeid'].index[prop_data['architecturalstyletypeid'].apply(lambda x: x in {'27', '5'})]] = np.nan

    # area_base_finished, missing rate 0.9995, dropped
    RENAMING_MAP['basementsqft'] = 'area_base_finished'

    # num_bathroom_assessor, no missing in training, few missing in prop
    RENAMING_MAP['bathroomcnt'] = 'num_bathroom_assessor'
    KEEP_FEATURE.add('bathroomcnt')
    prop_data['bathroomcnt'].fillna(prop_data['bathroomcnt'].mode().values[0], inplace=True)  # training no missing data, so must fill missing for prop

    # num_bathroom_zillow, missing rate 0.0131
    RENAMING_MAP['calculatedbathnbr'] = 'num_bathroom_zillow'
    KEEP_FEATURE.add('calculatedbathnbr')

    # num_bedroom, no missing in training, few missing in prop
    RENAMING_MAP['bedroomcnt'] = 'num_bedroom'
    KEEP_FEATURE.add('bedroomcnt')
    prop_data['bedroomcnt'].fillna(prop_data['bedroomcnt'].mode().values[0], inplace=True)

    # type_building_framing, missing rate 0.9998, dropped
    RENAMING_MAP['buildingclasstypeid'] = 'type_building_framing'
    cat_num_to_str('buildingclasstypeid')
    prop_data['buildingclasstypeid'].loc[prop_data['buildingclasstypeid'].index[prop_data['buildingclasstypeid'] == '4']] = np.nan

    # rank_building_quality, missing rate 0.3646
    RENAMING_MAP['buildingqualitytypeid'] = 'rank_building_quality'
    KEEP_FEATURE.add('buildingqualitytypeid')

    # type_deck, missing rate 0.9927, dropped
    RENAMING_MAP['decktypeid'] = 'type_deck'
    cat_num_to_str('decktypeid')

    # area_firstfloor_zillow, missing rate 0.9241, dropped
    RENAMING_MAP['finishedfloor1squarefeet'] = 'area_firstfloor_zillow'

    # area_living_finished_calc, missing rate 0.0073
    RENAMING_MAP['calculatedfinishedsquarefeet'] = 'area_living_finished_calc'
    KEEP_FEATURE.add('calculatedfinishedsquarefeet')

    # area_living_type_12, missing rate 0.0518
    RENAMING_MAP['finishedsquarefeet12'] = 'area_living_type_12'
    KEEP_FEATURE.add('finishedsquarefeet12')

    # area_living_type_13, missing rate 0.9996, dropped
    RENAMING_MAP['finishedsquarefeet13'] = 'area_living_type_13'

    # area_living_type_15, missing rate 0.9605, dropped
    RENAMING_MAP['finishedsquarefeet15'] = 'area_living_type_15'

    # area_firstfloor_assessor, missing rate 0.9241, dropped
    RENAMING_MAP['finishedsquarefeet50'] = 'area_firstfloor_assessor'

    # area_living_type_6, missing rate 0.9953, dropped
    RENAMING_MAP['finishedsquarefeet6'] = 'area_living_type_6'

    # code_ips, no missing in train, few missing in prop
    RENAMING_MAP['fips'] = 'code_ips'
    cat_num_to_str('fips')
    KEEP_FEATURE.add('fips')
    prop_data['fips'].fillna(prop_data['fips'].mode().values[0], inplace=True)

    # num_fireplace, directly impute nan as 0, NOTE, first leave it as nan, thus no missing
    RENAMING_MAP['fireplacecnt'] = 'num_fireplace'
    # prop_data['fireplacecnt'].fillna(0, inplace=True)
    # train_data['fireplacecnt'].fillna(0, inplace=True)
    KEEP_FEATURE.add('fireplacecnt')

    # num_fullbath, impute from bathroom cnt, first leave it as nan. bathroom no missing, thus no missing after imputation
    RENAMING_MAP['fullbathcnt'] = 'num_fullbath'

    def nan_impute_fullbathcnt(data):
        """nan impute fullbathcnt from bathroomcnt"""
        null_idx = data.index[data['fullbathcnt'].isnull()]
        fill_val = data['bathroomcnt'][null_idx].copy()
        fill_val_raw = fill_val.copy()
        fill_val_raw_floor = fill_val_raw.apply(math.floor)
        int_idx = np.abs(fill_val_raw.values - fill_val_raw_floor.values) < 1e-12
        fill_val[int_idx] = fill_val_raw[int_idx] - 1
        fill_val[~int_idx] = fill_val_raw_floor[~int_idx]
        data.loc[null_idx, 'fullbathcnt'] = fill_val
        return data

    # prop_data = nan_impute_fullbathcnt(prop_data)
    # train_data = nan_impute_fullbathcnt(train_data)
    KEEP_FEATURE.add('fullbathcnt')

    # num_garage, missing rate 0.6684, keep for now
    RENAMING_MAP['garagecarcnt'] = 'num_garage'
    KEEP_FEATURE.add('garagecarcnt')

    # area_garage, mark zero-area but non-zero num as nan, leave it as it is first. missing rate 0.7672
    RENAMING_MAP['garagetotalsqft'] = 'area_garage'

    def zero_impute_area_garage(data):
        target_idx = data.index[np.abs(data['garagetotalsqft'] - 0) < 1e-12]
        nan_idx = target_idx[data.loc[target_idx, 'garagecarcnt'] > 0]
        data.loc[nan_idx, 'garagetotalsqft'] = np.nan

    # zero_impute_area_garage(prop_data)
    # zero_impute_area_garage(train_data)
    KEEP_FEATURE.add('garagetotalsqft')

    # flag_spa_zillow, True and False, only True in data, mark nan as False, so no missing rate
    RENAMING_MAP['hashottuborspa'] = 'flag_spa_zillow'
    prop_data.loc[prop_data.index[prop_data['hashottuborspa'].isnull()], 'hashottuborspa'] = 'FALSE'
    train_data.loc[train_data.index[train_data['hashottuborspa'].isnull()], 'hashottuborspa'] = 'FALSE'
    KEEP_FEATURE.add('hashottuborspa')

def property_cleaning_kaggle():
    """for real kaggle submission, considering:
       1, outlier detection.
       2, missing data imputation from prop data.
       3, categorical variables group combining."""
    pass


def prop_imputation(col_name):
    pass




