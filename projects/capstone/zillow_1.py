import numpy as np
import pandas as pd
import math
import gc
from time import time
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split


feature_info = pd.read_csv('data/feature_info.csv', index_col='orig_name')
nmap_orig_to_new =  dict(zip(feature_info.index.values, feature_info['new_name'].values))
nmap_new_to_orig =  dict(zip(feature_info['new_name'].values, feature_info.index.values))

feature_imp_naive_lgb_split = pd.read_csv('records/feature_importance_split_naive_lgb.csv')
feature_imp_naive_lgb_gain = pd.read_csv('records/feature_importance_gain_naive_lgb.csv')


def cv_mean_model(train_y):
    np.random.seed(42)
    y = np.array(train_y)
    np.random.shuffle(y)
    n_fold = 5  # same as in lightGBM.cv
    fold_size = int(len(y) / n_fold) + 1

    evals = np.zeros(n_fold)
    for n in range(n_fold):
        if n == 0:
            train_data = y[fold_size:]
            test_data = y[:fold_size]
        elif n == n_fold - 1:
            train_data = y[-fold_size:]
            test_data = y[:-fold_size]
        else:
            train_data = np.r_[y[: fold_size * n], y[fold_size * (n + 1):]]
            test_data = y[fold_size * n: fold_size * (n + 1)]
        estimate = np.mean(train_data)
        evals[n] = np.mean(np.abs(test_data - estimate))
    return np.mean(evals)


def cv_meta_model(train_x, train_y, model, pre_applyfunc=None, post_applyfuc=None):
    """cv for applyfunc, transformation of inp data or out data in addition to applying model, cannot do with built-in cv.
       for now, pre_applyfunc is designed for row outlier cleaning.
       post_applyfunc is for seasonality handling."""
    pass


def cat_num_to_str_inner(data, col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not data[col_name].dtype == 'O':
        data.loc[:, col_name] = data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def property_cleaning_base(train_data, test_data):
    """basic feature clearning"""

    def cat_num_to_str(col_name):
        """for numeric-like categorical varible, transform to string, keep nan"""
        cat_num_to_str_inner(train_data, col_name)
        cat_num_to_str_inner(test_data, col_name)

    def mark_flag_col(col_name):
        """mark bool for numerical columns, mark True for val > 0, and False otherwise (include NaN)"""
        test_data_marks_true = test_data[col_name] >= 0.5
        test_data.loc[test_data.index[test_data_marks_true], col_name] = 'TRUE'
        test_data.loc[test_data.index[~test_data_marks_true], col_name] = 'FALSE'
        train_data_marks_true = train_data[col_name] >= 0.5
        train_data.loc[train_data.index[train_data_marks_true], col_name] = 'TRUE'
        train_data.loc[train_data.index[~train_data_marks_true], col_name] = 'FALSE'

    def mark_flag_col_tax_delinquency():
        test_data_marks_true = test_data['taxdelinquencyflag'] == 'Y'
        test_data.loc[test_data.index[test_data_marks_true], 'taxdelinquencyflag'] = 'TRUE'
        test_data.loc[test_data.index[~test_data_marks_true], 'taxdelinquencyflag'] = 'FALSE'
        train_data_marks_true = train_data['taxdelinquencyflag'] == 'Y'
        train_data.loc[train_data.index[train_data_marks_true], 'taxdelinquencyflag'] = 'TRUE'
        train_data.loc[train_data.index[~train_data_marks_true], 'taxdelinquencyflag'] = 'FALSE'

    def col_fill_na(data, col, fill):
        if fill == 'mode':
            data[col].fillna(data[col].mode().values[0], inplace=True)
        elif fill == 'mean':
            data[col].fillna(data[col].mean(), inplace=True)
        elif fill == '0':
            data[col].fillna(0, inplace=True)
        else:
            print('unknown fill type: %s' % fill)

    def clear_cat_col_group(data, col, groups):
        """set given groups of a categorical col to na"""
        in_group_flag = data[col].apply(lambda x: x in groups)
        data[col].loc[data[col].index[in_group_flag]] = np.nan

    # special treatment functions
    def raw_census_info_split():
        # create new categorical columns 'raw_census', 'raw_block'
        def raw_census_info_split_inner(data):
            data['temp'] = data['rawcensustractandblock'].apply(
                lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
            data.loc[:, 'temp'] = data['temp'].astype('O')
            data['raw_census'] = data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
            data['raw_block'] = data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
            data.drop('temp', axis=1, inplace=True)

        raw_census_info_split_inner(test_data)
        raw_census_info_split_inner(train_data)

    def census_info_split():
        # create new categorical columns 'raw_census', 'raw_block'
        def census_info_split_inner(data):
            data['census'] = data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
            data['block'] = data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)

        test_data['censustractandblock'] = test_data['censustractandblock'].apply(
            lambda x: str(int(x)) if not np.isnan(x) else 'nan')
        train_data['censustractandblock'] = train_data['censustractandblock'].apply(
            lambda x: str(int(x)) if not np.isnan(x) else 'nan')
        census_info_split_inner(test_data)
        census_info_split_inner(train_data)

    def nan_impute_fullbathcnt():
        """nan impute fullbathcnt from bathroomcnt(which is no missing)"""

        def nan_impute_fullbathcnt_inner(data):
            null_idx = data.index[data['fullbathcnt'].isnull()]
            fill_val = data['bathroomcnt'][null_idx].copy()
            fill_val_raw = fill_val.copy()
            fill_val_raw_floor = fill_val_raw.apply(math.floor)
            int_idx = np.abs(fill_val_raw.values - fill_val_raw_floor.values) < 1e-12
            fill_val[int_idx] = np.maximum(fill_val_raw[int_idx] - 1, 0)
            fill_val[~int_idx] = fill_val_raw_floor[~int_idx]
            data.loc[null_idx, 'fullbathcnt'] = fill_val
            return data

        nan_impute_fullbathcnt_inner(test_data)
        nan_impute_fullbathcnt_inner(train_data)

    def garage_area_cleaning():
        """clearn garage features, with zero area but sample has non-zero count"""

        def zero_impute_area_garage_inner(data, data_name):
            nan_idx = data.index[np.logical_and(np.abs(data['garagetotalsqft'] - 0) < 1e-12, data['garagecarcnt'] > 0)]
            data.loc[nan_idx, 'garagetotalsqft'] = np.nan  # to do better to impute with cnt group mean
            print('cleaned rows for %s: %d' % data_name, len(nan_idx))

        zero_impute_area_garage_inner(test_data, 'test_data')
        zero_impute_area_garage_inner(train_data, 'train_data')

    # type_air_conditioning
    cat_num_to_str('airconditioningtypeid')
    clear_cat_col_group(test_data, 'airconditioningtypeid', ['12'])

    # type_architectural_style
    cat_num_to_str('architecturalstyletypeid')
    clear_cat_col_group(test_data, 'architecturalstyletypeid', ['27', '5'])

    # area_base_finished

    # num_bathroom_assessor
    col_fill_na(test_data, 'bathroomcnt', 'mode')

    # num_bathroom_zillow

    # num_bedroom
    col_fill_na(test_data, 'bedroomcnt', 'mode')

    # type_building_framing
    cat_num_to_str('buildingclasstypeid')
    clear_cat_col_group(test_data, 'buildingclasstypeid', ['4'])

    # rank_building_quality

    # type_deck
    cat_num_to_str('decktypeid')

    # area_firstfloor_zillow

    # area_living_finished_calc

    # area_living_type_12

    # area_living_type_13

    # area_living_type_15

    # area_firstfloor_assessor

    # area_living_type_6

    # code_ips, no missing in train
    cat_num_to_str('fips')
    col_fill_na(test_data, 'fips', 'mode')

    # num_fireplace
    # col_fill_na(test_data, 'fireplacecnt', '0')

    # num_fullbath

    # num_garage

    # area_garage

    # flag_spa_zillow
    mark_flag_col('hashottuborspa')

    # type_heating_system
    cat_num_to_str('heatingorsystemtypeid')
    clear_cat_col_group(test_data, 'heatingorsystemtypeid', ['19', '21'])

    # latitude
    col_fill_na(test_data, 'latitude', 'mean')

    # longitude
    col_fill_na(test_data, 'longitude', 'mean')

    # area_lot

    # flag_pool
    mark_flag_col('poolcnt')

    # area_pool
    test_data.loc[test_data.index[test_data['poolcnt'] == 'FALSE'], 'poolsizesum'] = 0
    train_data.loc[train_data.index[train_data['poolcnt'] == 'FALSE'], 'poolsizesum'] = 0

    # pooltypeid10, high missing rate, counter intuitive values. drop it
    mark_flag_col('pooltypeid10')
    
    # pooltypeid2 and pooltypeid7
    mark_flag_col('pooltypeid2')
    mark_flag_col('pooltypeid7')

    def make_pool_type(data):
        data['type_pool'] = 'None'
        data.loc[data.index[data['pooltypeid2'] == 'TRUE'], 'type_pool'] = 'TRUE'
        data.loc[data.index[data['pooltypeid7'] == 'TRUE'], 'type_pool'] = 'FALSE'

    make_pool_type(test_data)
    make_pool_type(train_data)

    # code_county_landuse
    col_fill_na(test_data, 'propertycountylandusecode', 'mode')

    # code_county_landuse
    cat_num_to_str('propertylandusetypeid')
    clear_cat_col_group(test_data, 'propertylandusetypeid', ['270'])
    col_fill_na(test_data, 'propertylandusetypeid', 'mode')

    # str_zoning_desc

    # raw_census_block, raw_census, raw_block.
    raw_census_info_split()
    col_fill_na(test_data, 'raw_census', 'mode')
    col_fill_na(test_data, 'raw_block', 'mode')

    # code_city
    cat_num_to_str('regionidcity')

    # code_county
    cat_num_to_str('regionidcounty')

    # code_neighborhood
    cat_num_to_str( 'regionidneighborhood')

    # code_zip
    cat_num_to_str('regionidzip')

    # num_room
    col_fill_na(test_data, 'roomcnt', 'mode')

    # type_story
    cat_num_to_str('storytypeid')

    # num_34_bathroom
    cat_num_to_str('threequarterbathnbr')

    # type_construction
    clear_cat_col_group(test_data, 'typeconstructiontypeid', ['2'])
    cat_num_to_str('typeconstructiontypeid')

    # num_unit

    # area_yard_patio

    # area_yard_storage

    # year_built

    # num_story
    cat_num_to_str('numberofstories')

    # flag_fireplace
    mark_flag_col('fireplaceflag')

    # dollar_taxvalue_structure

    # dollar_taxvalue_total
    col_fill_na(test_data, 'taxvaluedollarcnt', 'mean')

    # dollar_taxvalue_land
    col_fill_na(test_data, 'landtaxvaluedollarcnt', 'mean')

    # dollar_tax

    # flag_tax_delinquency
    mark_flag_col_tax_delinquency()

    # year_tax_due
    cat_num_to_str('taxdelinquencyyear')

    # census_block
    census_info_split()


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


def load_data_raw():
    # init load data
    prop_data = pd.read_csv('data/properties_2016.csv', header=0)
    error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
    error_data['sale_month'] = error_data['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month

    train_data = error_data.merge(prop_data, how='left', on='parcelid')
    # train_data.to_csv('data/train_data_merge.csv', index=False)

    submission = pd.read_csv('data/sample_submission.csv', header=0)
    submission['parcelid'] = submission['ParcelId']
    submission.drop('ParcelId', axis=1, inplace=True)
    test_data = submission.merge(prop_data, how='left', on='parcelid')
    property_cleaning_base(train_data, test_data)

    return train_data, test_data


def convert_cat_col(train_data, test_data, col):
    """version for production training & testing, make sure both data sets uses the same value map"""
    # convert to lgb usable categorical
    # set nan to string so that can be sorted, make sure training set and testing set get the same coding
    test_data.loc[test_data.index[test_data[col].isnull()], col] = 'nan'
    train_data.loc[train_data.index[train_data[col].isnull()], col] = 'nan'

    # create and use map from test data only
    uni_vals = np.sort(test_data[col].unique()).tolist()

    m = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
    train_data.loc[:, col] = train_data[col].apply(lambda x: m[x])
    train_data.loc[:, col] = train_data[col].astype('category')

    test_data.loc[:, col] = test_data[col].apply(lambda x: m[x])
    test_data.loc[:, col] = test_data[col].astype('category')


def convert_cat_col_single(data, col):
    # convert to lgb usable categorical
    # set nan to string so that can be sorted, make sure training set and testing set get the same coding
    data.loc[data.index[data[col].isnull()], col] = 'nan'
    uni_vals = np.sort(data[col].unique()).tolist()
    map = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
    data.loc[:, col] = data[col].apply(lambda x: map[x])
    data.loc[:, col] = data[col].astype('category')


def load_data_naive_lgb(train_data, test_data):
    keep_feature = list(feature_info.index[feature_info['class'].apply(lambda x: True if x in {1, 2} else False)].values)
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    def float_type_cast(data):
        for col in data.columns:
            if data[col].dtype == np.float64:
                data.loc[:, col] = data[col].astype(np.float32)

    # type clearning
    float_type_cast(test_x)
    float_type_cast(train_x)
    for col in set(train_x.columns).intersection(set(test_x.columns)):
        if feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col(train_x, test_x, col)

    return test_x, train_x, train_y


def load_data_naive_lgb_feature_up(train_data, test_data):
    # load features
    keep_feature = list(feature_info.index[feature_info['class'].apply(lambda x: True if x in {1, 2, 4} else False)].values)
    for f in ['year_assess', 'census_block', 'raw_census_block']:
        # we have multiple year_assess values in test data, but only one value in train data
        # census_block raw info is not valid variable.
        if nmap_new_to_orig[f] in keep_feature:
            keep_feature.remove(nmap_new_to_orig[f])
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    def float_type_cast(data):
        for col in data.columns:
            if data[col].dtype == np.float64:
                data.loc[:, col] = data[col].astype(np.float32)

    # type clearning
    float_type_cast(test_x)
    float_type_cast(train_x)
    for col in set(train_x.columns).intersection(set(test_x.columns)):
        if feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col(train_x, test_x, col)

    return test_x, train_x, train_y


def load_data_naive_lgb_feature_down(train_data, test_data):
    """use subset of good features to see how it performance"""
    keep_feature = list(feature_info.index[feature_info['class'].apply(lambda x: True if x == 1 else False)].values)
    keep_feature += [nmap_new_to_orig[n] for n in ['area_garage',
                                                   'rank_building_quality',
                                                   'area_pool',
                                                   'num_bathroom_assessor',
                                                   'num_unit']]
    for f in ['code_fips', 'num_fullbath', 'flag_spa_zillow', 'flag_tax_delinquency']:
        keep_feature.remove(nmap_new_to_orig[f])

    if len(keep_feature) != len(set(keep_feature)):
        raise Exception('duplicated feature')
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    def float_type_cast(data):
        for col in data.columns:
            if data[col].dtype == np.float64:
                data.loc[:, col] = data[col].astype(np.float32)

    # type clearning
    float_type_cast(test_x)
    float_type_cast(train_x)
    for col in set(train_x.columns).intersection(set(test_x.columns)):
        if feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col(train_x, test_x, col)

    return test_x, train_x, train_y


def load_data_naive_lgb_final(train_data, test_data):
    """use subset of good features to see how it performance"""
    keep_feature = list(feature_info.index[feature_info['class'].apply(lambda x: True if x in {1, 2} else False)].values)
    keep_feature += [nmap_new_to_orig[n] for n in ['area_living_type_15',
                                                   'area_firstfloor_zillow',
                                                   'area_firstfloor_assessor',
                                                   'area_yard_patio',
                                                   'year_tax_due']]

    if len(keep_feature) != len(set(keep_feature)):
        raise Exception('duplicated feature')
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    def float_type_cast(data):
        for col in data.columns:
            if data[col].dtype == np.float64:
                data.loc[:, col] = data[col].astype(np.float32)

    # type clearning
    float_type_cast(test_x)
    float_type_cast(train_x)
    for col in set(train_x.columns).intersection(set(test_x.columns)):
        if feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col(train_x, test_x, col)

    return test_x, train_x, train_y


def train_lgb_with_val(train_x, train_y):
    # train - validaiton split
    train_x_use, val_x, train_y_use, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

    # create lgb dataset
    lgb_train = lgb.Dataset(train_x_use, train_y_use)
    lgb_val = lgb.Dataset(val_x, val_y, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': {'l1'},
        'num_leaves': 40,
        'min_data_in_leaf': 300,
        'learning_rate': 0.01,
        'lambda_l2': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbosity': 0
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=2000,
                    valid_sets=lgb_val,
                    early_stopping_rounds=30)

    return gbm


def train_lgb(train_x, train_y, num_boost_round=1500):

    # create lgb dataset
    lgb_train = lgb.Dataset(train_x, train_y)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': {'l1'},
        'num_leaves': 40,
        'min_data_in_leaf': 300,
        'learning_rate': 0.01,
        'lambda_l2': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbosity': 0
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=num_boost_round)

    return gbm


def cv_lgb_final(train_x, train_y, num_boosting_round=1500):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': {'l1'},
        'num_leaves': 40,
        'min_data_in_leaf': 300,
        'learning_rate': 0.01,
        'lambda_l2': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbosity': 0
    }
    eval_hist = lgb.cv(params, lgb_train, num_boost_round=num_boosting_round)
    return eval_hist['l1-mean'][-1]


def feature_importance(gbm, label=None):
    """only applicable for niave gbm"""
    features = np.array(gbm.feature_name())
    feature_info = pd.read_csv('data/feature_info.csv', index_col='orig_name')

    importance_split = gbm.feature_importance('split')
    sort_split = np.argsort(importance_split)
    sort_split = sort_split[::-1]
    features_split_sort = features[sort_split]
    features_name_new_split = [feature_info.loc[f, 'new_name'] if f in feature_info.index else f for f in features_split_sort]
    features_class_split = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_split_sort]
    importance_split_sort = importance_split[sort_split]

    importance_gain = gbm.feature_importance('gain')
    sort_gain = np.argsort(importance_gain)
    sort_gain = sort_gain[::-1]
    features_gain_sort = features[sort_gain]
    features_name_new_gain = [feature_info.loc[f, 'new_name'] if f in feature_info.index else f for f in features_gain_sort]
    features_class_gain = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_gain_sort]
    importance_gain_sort = importance_gain[sort_gain]

    df_split = pd.DataFrame({'feature': features_name_new_split, 'split': importance_split_sort, 'class': features_class_split})
    df_gain = pd.DataFrame({'feature': features_name_new_gain, 'gain': importance_gain_sort, 'class': features_class_gain})

    if label:
        print(df_split)
        print(df_gain)

        df_split.to_csv('records/feature_importance_split_%s.csv' % label, index=False)
        df_gain.to_csv('records/feature_importance_gain_%s.csv' % label, index=False)
    return df_split, df_gain


def feature_importance_rank(gbm, col_inp, col_ref=None):
    """returns the rank (avg of split and gain) of given feature, input col should be new-name-convention.
       if col_ref is provided, also output rank for col_ref, for comparison"""
    col_inp_orig = nmap_new_to_orig[col_inp] if col_inp in nmap_new_to_orig else col_inp
    col_ref_orig = nmap_new_to_orig[col_ref] if col_ref in nmap_new_to_orig else col_ref

    features = gbm.feature_name()
    n_features = len(features)
    print('n_features total: %d' % n_features)

    def feature_rank_importance_inner(gbm, col):
        features = gbm.feature_name()
        feature_idx = features.index(col)
        # use reverse order in rank
        rank_feature_split = n_features - np.argsort(np.argsort(gbm.feature_importance('split')))[feature_idx]
        rank_feature_gain = n_features - np.argsort(np.argsort(gbm.feature_importance('gain')))[feature_idx]
        return rank_feature_split, rank_feature_gain, (rank_feature_split + rank_feature_gain) / 2

    rank_inp_split, rank_inp_gain, rank_inp_avg = feature_rank_importance_inner(gbm, col_inp_orig)
    print('%s: rank split = %d, rank gain = %d, avg_rank = %.1f' % (col_inp, rank_inp_split, rank_inp_gain, rank_inp_avg))
    if col_ref:
        rank_ref_split = list(feature_imp_naive_lgb_split['feature']).index(col_ref) + 1
        rank_ref_gain = list(feature_imp_naive_lgb_gain['feature']).index(col_ref) + 1
        rank_ref_avg = (rank_ref_split + rank_ref_gain) / 2
        print('%s: rank split = %d, rank gain = %d, avg_rank = %.1f' % (col_ref, rank_ref_split, rank_ref_gain, rank_ref_avg))


def search_lgb_random(train_x, train_y):
    lgb_train = lgb.Dataset(train_x, train_y)

    def rand_min_data_in_leaf():
        return np.random.randint(100, 500)

    def rand_learning_rate():
        return np.random.uniform(0.008, 0.012)

    def rand_num_leaf():
        return np.random.randint(30, 40)

    def rand_lambda_l2():
        return np.random.uniform(0.01, 0.04)

    n_trail = 100
    res = []
    for i in range(1, n_trail + 1):
        rand_params = {'num_leaves': rand_num_leaf(),
                       'min_data_in_leaf': rand_min_data_in_leaf(),
                       'learning_rate': rand_learning_rate(),
                       'lambda_l2': rand_lambda_l2()}
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression_l1',
            'metric': {'l1'},
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'verbosity': 0
        }
        params.update(rand_params)
        eval_hist = lgb.cv(params, lgb_train, num_boost_round=2000, early_stopping_rounds=30)
        res.append([eval_hist['l1-mean'][-1],
                    rand_params['num_leaves'],
                    rand_params['min_data_in_leaf'],
                    rand_params['learning_rate'],
                    rand_params['lambda_l2']])
        print('finished %d / %d' % (i, n_trail))
    res_df = pd.DataFrame(res, columns=['score', 'num_leaves', 'min_data_in_leaf', 'learning_rate', 'lambda_l2'])
    res_df.to_csv('temp_cv_res_random.csv', index=False)


def search_lgb_grid(train_x, train_y):
    lgb_train = lgb.Dataset(train_x, train_y)

    min_data_in_leaf_list = []
    learning_rate_list = []
    num_leaf_list = []
    lambda_l2_list = []

    res = []
    n_trial = len(min_data_in_leaf_list) * len(learning_rate_list) * len(num_leaf_list) * len(lambda_l2_list)
    iter = 0
    for min_data_in_leaf in min_data_in_leaf_list:
        for learning_rate in learning_rate_list:
            for num_leaf in num_leaf_list:
                for lambda_l2 in lambda_l2_list:
                    params = {
                        'boosting_type': 'gbdt',
                        'objective': 'regression_l1',
                        'metric': {'l1'},
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.7,
                        'bagging_freq': 5,
                        'verbosity': 0,
                        'num_leaves': num_leaf,
                        'min_data_in_leaf': min_data_in_leaf,
                        'learning_rate': learning_rate,
                        'lambda_l2': lambda_l2
                    }
                    eval_hist = lgb.cv(params, lgb_train, num_boost_round=2000, early_stopping_rounds=10)
                    res.append([eval_hist['l1-mean'][-1],
                                num_leaf,
                                min_data_in_leaf,
                                learning_rate,
                                lambda_l2])
                    iter += 1
                    print('finished %d / %d' % (iter, n_trial))
    res_df = pd.DataFrame(res, columns=['score', 'num_leaves', 'min_data_in_leaf', 'learning_rate', 'lambda_l2'])
    res_df.to_csv('temp_cv_res_grid.csv', index=False)


def submit_nosea(score, index, ver):
    df = pd.DataFrame()
    df['ParcelId'] = index
    for col in ('201610', '201611', '201612', '201710', '201711', '201712'):
        df[col] = score
    date_str = ''.join(str(datetime.date.today()).split('-'))
    print(df.shape)
    df.to_csv('data/submission_%s_v%d.csv.gz' % (date_str, ver), index=False, float_format='%.4f', compression='gzip')


def pred_nosea(model, test_x):
    train_data, test_data = load_data_raw()
    test_x, train_x, train_y = load_data_naive_lgb(train_data, test_data)
    trained_gbm = train_lgb(train_x, train_y)
    pred_score = trained_gbm.predict(test_x)
    submit_nosea(pred_score, test_data['parcelid'], 2)


def feature_engineering(train_x_inp, train_data, test_x_inp, test_data):

    def feature_engineering_inner(target_data, raw_data):
        """target_data are those used for model input, it is output from naive lgb selection, thus does not contain all features"""
        # number_full_bath
        orig_name_fullbath = nmap_new_to_orig['num_fullbath']
        orig_name_bath = nmap_new_to_orig['num_bathroom_assessor']
        # first impute nan
        null_idx = raw_data.index[raw_data[orig_name_fullbath].isnull()]
        fill_val = raw_data[orig_name_bath][null_idx].copy()
        fill_val_floor = fill_val.apply(math.floor)
        int_idx = np.abs(fill_val.values - fill_val_floor.values) < 1e-12
        fill_val[int_idx] = np.maximum(fill_val[int_idx] - 1, 0)
        fill_val[~int_idx] = fill_val_floor[~int_idx]
        target_data['num_fullbath_impute'] = raw_data[orig_name_fullbath]
        target_data.loc[null_idx, 'num_fullbath_impute'] = fill_val
        # then group types
        target_data['num_fullbath_clean'] = target_data['num_fullbath_impute']
        target_data.loc[target_data.index[target_data['num_fullbath_clean'] >= 6], 'num_fullbath_clean'] = 6
        cat_num_to_str_inner(target_data, 'num_fullbath_clean')
        target_data.drop('num_fullbath_impute', axis=1, inplace=True)

        # dollar_taxvalue variables
        orig_name_structrue = nmap_new_to_orig['dollar_taxvalue_structure']
        orig_name_land = nmap_new_to_orig['dollar_taxvalue_land']
        orig_name_total = nmap_new_to_orig['dollar_taxvalue_total']
        target_data['dollar_taxvalue_structure_land_diff'] = raw_data[orig_name_structrue] - raw_data[orig_name_land]
        target_data['dollar_taxvalue_structure_land_absdiff'] = np.abs(raw_data[orig_name_structrue] - raw_data[orig_name_land])
        target_data['dollar_taxvalue_structure_total_ratio'] = raw_data[orig_name_structrue] / raw_data[orig_name_total]
        target_data['dollar_taxvalue_structure_land_ratio'] = raw_data[orig_name_structrue] / raw_data[orig_name_land]
        target_data['dollar_taxvalue_total_structure_ratio'] = raw_data[orig_name_total] / raw_data[orig_name_structrue]
        target_data['dollar_taxvalue_total_land_ratio'] = raw_data[orig_name_total] / raw_data[orig_name_land]
        target_data['dollar_taxvalue_land_structure_ratio'] = raw_data[orig_name_land] / raw_data[orig_name_structrue]

    test_x = test_x_inp.copy()
    train_x = train_x_inp.copy()
    feature_engineering_inner(test_x, test_data)
    feature_engineering_inner(train_x, train_data)
    convert_cat_col(train_x, test_x, 'num_fullbath_clean')
    # class 3 variables
    for name in ('code_city', 'code_neighborhood', 'code_zip', 'raw_block', 'block', 'str_zoning_desc', 'code_county_landuse'):
        used_name = name + '_orig'  # not directly use original name here as naive lgb assumes all potential original cols have been used
        train_x[used_name] = train_data[nmap_new_to_orig[name]]
        test_x[used_name] = test_data[nmap_new_to_orig[name]]
        convert_cat_col(train_x, test_x, used_name)
    return train_x, test_x









