import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import math
import gc
import os
from time import time
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# from bayes_opt import BayesianOptimization
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

warnings.filterwarnings('ignore', category=DeprecationWarning)


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')
nmap_orig_to_new =  dict(zip(feature_info['orig_name'].values, feature_info.index.values))
nmap_new_to_orig =  dict(zip(feature_info.index.values, feature_info['orig_name'].values))

feature_imp_naive_lgb = pd.read_csv('records/feature_importance_raw_all.csv')

# params_naive = {
#     'boosting_type': 'gbdt',
#     'objective': 'regression_l1',
#     'metric': {'l1'},
#     'num_leaves': 40,
#     'min_data_in_leaf': 300,
#     'learning_rate': 0.01,
#     'lambda_l2': 0.02,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 5,
#     'verbosity': 0,
#     'num_boosting_rounds': 1500
# }

params_naive = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': {'l1'},
    'num_leaves': 48,
    'min_data_in_leaf': 200,
    'learning_rate': 0.0045,
    'lambda_l2': 0.004,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbosity': 0,
    'num_boosting_rounds': 2200
}


params_fe = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': {'l1'},
    'num_leaves': 40,
    'min_data_in_leaf': 200,
    'learning_rate': 0.005,
    'lambda_l2': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbosity': 0,
    'num_boosting_rounds': 2000
}


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


def cv_meta_model(x, y, test_data, model_func, outlier_thresh=0.001, outlier_handling=None):
    """cv for applyfunc, transformation of inp data or out data in addition to applying model, cannot do with built-in cv.
       for now, pre_applyfunc is designed for row outlier cleaning.
       post_applyfunc is for seasonality handling.
       expected input format, y being a Series, x being a DataFrame"""
    np.random.seed(42)
    idx = np.array(range(y.shape[0]))
    np.random.shuffle(idx)
    n_fold = 5  # same as in lightGBM.cv
    fold_size = int(y.shape[0] / n_fold) + 1

    evals = np.zeros(n_fold)
    for n in range(n_fold):
        if n == 0:
            train_x_cv, train_y_cv = x.iloc[idx[fold_size:], :].copy(), y.iloc[idx[fold_size:]].copy()
            test_x_cv, test_y_cv = x.iloc[idx[:fold_size], :].copy(), y.iloc[idx[:fold_size]].copy()
        elif n == n_fold - 1:
            train_x_cv, train_y_cv = x.iloc[idx[:-fold_size], :].copy(), y.iloc[idx[:-fold_size]].copy()
            test_x_cv, test_y_cv = x.iloc[idx[-fold_size:], :].copy(), y.iloc[idx[-fold_size:]].copy()
        else:
            train_x_cv = pd.concat([x.iloc[idx[: fold_size * n], :].copy(), x.iloc[idx[fold_size * (n + 1):], :].copy()])
            train_y_cv = pd.concat([y.iloc[idx[: fold_size * n]].copy(), y.iloc[idx[fold_size * (n + 1):]].copy()])
            test_x_cv = x.iloc[idx[fold_size * n: fold_size * (n + 1)], :].copy()
            test_y_cv = y.iloc[idx[fold_size * n: fold_size * (n + 1)]].copy()

        if outlier_handling:
            train_x_cv, train_y_cv = outlier_handling(train_x_cv, train_y_cv, test_data, outlier_thresh)
        pred_y_cv = model_func(train_x_cv, train_y_cv, test_x_cv)
        evals[n] = np.mean(np.abs(pred_y_cv - test_y_cv))
    return np.mean(evals)


def lgb_2step(train_x, train_y, test_x):
    """2-step training and prediction, first make use of linear relationship between year_built and logerro, then use features to predict rest error"""
    x = train_x['year_built'].fillna(1955)
    y = train_y.astype(np.float32)
    x = x.astype(np.float32)
    x = x.reshape(x.shape[0], 1)
    lr = LinearRegression()
    lr.fit(x, y)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': {'l1'},
        'num_leaves': 30,
        'min_data_in_leaf': 250,
        'learning_rate': 0.01,
        'lambda_l2': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbosity': 0,
        'num_boosting_rounds': 1500
    }

    error2 = y - lr.predict(x)
    gbm = train_lgb(train_x, error2, params)

    test_x_yearbuilt = test_x['year_built'].fillna(1955).reshape(test_x.shape[0], 1)
    test_y = gbm.predict(test_x) + lr.predict(test_x_yearbuilt)
    return test_y


def lgb_raw(train_x, train_y, test_x):

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression_l1',
        'metric': {'l1'},
        'num_leaves': 30,
        'min_data_in_leaf': 250,
        'learning_rate': 0.01,
        'lambda_l2': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbosity': 0,
        'num_boosting_rounds': 1500
    }

    gbm = train_lgb(train_x, train_y, params)
    return gbm.predict(test_x)


def outlier_y_rm(train_x, train_y, test_data, thresh):
    quantile_cut = thresh
    down_thresh, up_thresh = train_y.quantile([quantile_cut, 1 - quantile_cut])
    pos_ex_y_idx = train_y > up_thresh
    neg_ex_y_idx = train_y < down_thresh
    ex_y_idx = np.logical_or(pos_ex_y_idx, neg_ex_y_idx)
    # remove outlier
    train_x = train_x.loc[~ex_y_idx, :]
    train_y = train_y[~ex_y_idx]
    return train_x, train_y


def outlier_y_capfloor(train_x, train_y, test_data, thresh):
    quantile_cut = thresh
    down_thresh, up_thresh = train_y.quantile([quantile_cut, 1 - quantile_cut])
    pos_ex_y_idx = train_y > up_thresh
    neg_ex_y_idx = train_y < down_thresh
    # cap_floor outlier
    train_y[pos_ex_y_idx] = up_thresh
    train_y[neg_ex_y_idx] = down_thresh
    return train_x, train_y


def outlier_x_clean(train_x_inp, train_y_inp, test_x, type_train='rm', type_test=None, thresh=0.001):
    """2 strategies for x in train: remove, set to NA. (cap / floor is not expected to help for a tree based model).
       2 strategies for test: set to NA, leave as it is. 
       For each numerical variables, need to consider to do one-side or 2-sided cleaning"""
    rm_idx_train = {}
    train_x = train_x_inp.copy()
    train_y = train_y_inp.copy()

    def proc_outlier_num(col, q_down, q_up):
        """use -1 if one side of data is not trimmed"""
        thresh_up = test_x[col].quantile(q_up) if q_up > 0 else np.inf  # Series.quantile has already considered NA
        thresh_down = test_x[col].quantile(q_down) if q_down > 0 else -np.inf

        pos_idx_train = train_x[col] > thresh_up
        neg_idx_train = train_x[col] < thresh_down
        pos_idx_test = test_x[col] > thresh_up
        neg_idx_test = test_x[col] < thresh_down
        idx_train = np.logical_or(pos_idx_train, neg_idx_train)
        idx_test = np.logical_or(pos_idx_test, neg_idx_test)
        rm_idx_train[col] = idx_train
        train_x.loc[idx_train, col] = np.nan
        if type_test == 'na':
            test_x.loc[idx_test, col] = np.nan

    def proc_outlier_cat(col):
        idx_train = train_x[col].apply(TYPE_CAR_CLEAN_MAP[col])
        idx_test = test_x[col].apply(TYPE_CAR_CLEAN_MAP[col])
        rm_idx_train[col] = idx_train
        train_x.loc[idx_train, col] = np.nan
        if type_test == 'na':
            test_x.loc[idx_test, col] = np.nan

    # year built use left side only
    proc_outlier_num('year_built', thresh, -1)

    # all others use two-sided
    for num_var in ('area_lot', 'dollar_tax', 'area_living_type_12', 'dollar_taxvalue_structure',
                    'area_living_finished_calc', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
                    'area_garage', 'area_living_type_15', 'area_pool'):
        proc_outlier_num(num_var, thresh, 1-thresh)

    for cat_var in TYPE_CAR_CLEAN_MAP:
        if cat_var in ('type_heating_system', 'type_landuse'):
            # type variables have already been coded to int in load_data_naive_lgb
            continue
        proc_outlier_cat(cat_var)

    if type_train == 'rm':
        rm_idx = np.logical_or.reduce(list(rm_idx_train.values()))
        print('n_rows to remove: %d' % np.sum(rm_idx))
        train_x = train_x.loc[train_x.index[~rm_idx], :]
        train_y = train_y[~rm_idx]

    return train_x, train_y


def outlier_rm_x(train_x, train_y, test_data, thresh):
    return outlier_x_clean(train_x, train_y, test_data, thresh=thresh)


def outlier_na_x(train_x, train_y, test_data, thresh):
    return outlier_x_clean(train_x, train_y, test_data, type_train='na', type_test='na', thresh=thresh)


def cat_num_to_str(data, col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not data[col_name].dtype == 'O':
        data.loc[:, col_name] = data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def property_cleaning(prop_data):
    """basic feature clearning, for num_ variables, use as categorical. 
       for categorical variables, group small categories"""

    def mark_flag_col(col_name):
        """mark bool for numerical columns, mark True for val > 0, and False otherwise (include NaN)"""
        data_marks_true = prop_data[col_name] >= 0.5
        prop_data.loc[prop_data.index[data_marks_true], col_name] = 'TRUE'
        prop_data.loc[prop_data.index[~data_marks_true], col_name] = 'FALSE'

    def mark_flag_col_tax_delinquency():
        test_data_marks_true = prop_data['taxdelinquencyflag'] == 'Y'
        prop_data.loc[prop_data.index[test_data_marks_true], 'taxdelinquencyflag'] = 'TRUE'
        prop_data.loc[prop_data.index[~test_data_marks_true], 'taxdelinquencyflag'] = 'FALSE'

    def clear_cat_col_group(col, groups):
        """set given groups of a categorical col to na"""
        in_group_flag = prop_data[col].apply(lambda x: x in groups)
        prop_data.loc[prop_data[col].index[in_group_flag], col] = np.nan

    def col_fill_na(col, fill_type):
        if fill_type == 'mode':
            prop_data[col].fillna(prop_data[col].mode().values[0], inplace=True)
        elif fill_type == 'mean':
            prop_data[col].fillna(prop_data[col].mean(), inplace=True)
        elif fill_type == '0':
            prop_data[col].fillna(0, inplace=True)
        else:
            raise Exception('unknown fill_type')

    # type_air_conditioning
    merge_idx = prop_data['airconditioningtypeid'].apply(lambda x: x in {5, 13, 11, 9, 3})
    prop_data.loc[prop_data.index[merge_idx], 'airconditioningtypeid'] = 13
    cat_num_to_str(prop_data, 'airconditioningtypeid')
    clear_cat_col_group('airconditioningtypeid', ['12'])

    # type_architectural_style
    cat_num_to_str(prop_data, 'architecturalstyletypeid')
    clear_cat_col_group('architecturalstyletypeid', ['27', '5'])

    # area_base_finished

    # num_bathroom_assessor
    col_fill_na('bathroomcnt', 'mode')

    # num_bathroom_zillow

    # num_bedroom
    col_fill_na('bedroomcnt', 'mode')

    # type_building_framing
    cat_num_to_str(prop_data, 'buildingclasstypeid')
    clear_cat_col_group('buildingclasstypeid', ['4'])

    # rank_building_quality

    # type_deck
    cat_num_to_str(prop_data, 'decktypeid')

    # area_firstfloor_zillow

    # area_living_finished_calc

    # area_living_type_12

    # area_living_type_13

    # area_living_type_15

    # area_firstfloor_assessor

    # area_living_type_6

    # code_ips, no missing in train
    cat_num_to_str(prop_data, 'fips')
    col_fill_na('fips', 'mode')

    # num_fireplace
    # col_fill_na(test_data, 'fireplacecnt', '0')

    # num_fullbath
    null_idx = prop_data.index[prop_data['fullbathcnt'].isnull()]
    fill_val = prop_data['bathroomcnt'][null_idx].copy()
    fill_val_floor = fill_val.apply(math.floor)
    int_idx = np.abs(fill_val.values - fill_val_floor.values) < 1e-12
    fill_val[int_idx] = np.maximum(fill_val[int_idx] - 1, 0)
    fill_val[~int_idx] = fill_val_floor[~int_idx]
    prop_data.loc[null_idx, 'fullbathcnt'] = fill_val

    # num_garage

    # area_garage
    # fill group mean by garagecarcnt for thoes with carcnt > 0 but sqft == 0
    mark_idx = prop_data.index[np.logical_and(np.abs(prop_data['garagetotalsqft'] - 0) < 1e-12, prop_data['garagecarcnt'] > 0)]
    sub_df = prop_data.loc[mark_idx, ['garagetotalsqft', 'garagecarcnt']]
    sub_df = sub_df.join(prop_data['garagetotalsqft'].groupby(prop_data['garagecarcnt']).mean(), on='garagecarcnt', rsuffix='_avg')
    prop_data.loc[mark_idx, 'garagetotalsqft'] = sub_df['garagetotalsqft_avg']

    # flag_spa_zillow
    mark_flag_col('hashottuborspa')

    # type_heating_system
    cat_num_to_str(prop_data, 'heatingorsystemtypeid')
    clear_cat_col_group('heatingorsystemtypeid', ['19', '21'])

    # latitude
    col_fill_na('latitude', 'mean')

    # longitude
    col_fill_na('longitude', 'mean')

    # area_lot

    # flag_pool
    mark_flag_col('poolcnt')

    # area_pool
    prop_data.loc[prop_data.index[prop_data['poolcnt'] == 'FALSE'], 'poolsizesum'] = 0

    # pooltypeid10, high missing rate, counter intuitive values. drop it
    mark_flag_col('pooltypeid10')
    
    # pooltypeid2 and pooltypeid7
    mark_flag_col('pooltypeid2')
    mark_flag_col('pooltypeid7')
    prop_data['type_pool'] = 'None'
    prop_data.loc[prop_data.index[prop_data['pooltypeid2'] == 'TRUE'], 'type_pool'] = 'TRUE'
    prop_data.loc[prop_data.index[prop_data['pooltypeid7'] == 'TRUE'], 'type_pool'] = 'FALSE'

    # code_county_landuse
    cat_num_to_str(prop_data, 'propertycountylandusecode')
    col_fill_na('propertycountylandusecode', 'mode')

    # code_county_landuse
    cat_num_to_str(prop_data, 'propertylandusetypeid')
    clear_cat_col_group('propertylandusetypeid', ['270'])
    col_fill_na('propertylandusetypeid', 'mode')

    # str_zoning_desc
    cat_num_to_str(prop_data, 'propertyzoningdesc')

    # raw_census_block, raw_census, raw_block.
    prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
    prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('temp', axis=1, inplace=True)
    col_fill_na('raw_census', 'mode')
    col_fill_na('raw_block', 'mode')
    prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

    # code_city
    cat_num_to_str(prop_data, 'regionidcity')

    # code_county
    cat_num_to_str(prop_data, 'regionidcounty')

    # code_neighborhood
    cat_num_to_str(prop_data, 'regionidneighborhood')

    # code_zip
    cat_num_to_str(prop_data, 'regionidzip')

    # num_room
    col_fill_na('roomcnt', 'mode')

    # type_story
    cat_num_to_str(prop_data, 'storytypeid')

    # num_34_bathroom

    # type_construction
    clear_cat_col_group('typeconstructiontypeid', ['2'])
    cat_num_to_str(prop_data, 'typeconstructiontypeid')

    # num_unit

    # area_yard_patio

    # area_yard_storage

    # year_built

    # num_story

    # flag_fireplace
    mark_flag_col('fireplaceflag')

    # dollar_taxvalue_structure

    # dollar_taxvalue_total
    col_fill_na('taxvaluedollarcnt', 'mean')

    # dollar_taxvalue_land
    col_fill_na('landtaxvaluedollarcnt', 'mean')

    # dollar_tax

    # flag_tax_delinquency
    mark_flag_col_tax_delinquency()

    # year_tax_due

    # census_block
    prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
    prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('censustractandblock', axis=1, inplace=True)
    col_fill_na('census', 'mode')
    col_fill_na('block', 'mode')

    # for name in feature_info.index.values:
    #     if name[:4] == 'num_':
    #         cat_num_to_str(prop_data, nmap_new_to_orig[name])
    #         feature_info.loc[name, 'type'] = 'cat'

    # name columns
    prop_data.rename(columns=nmap_orig_to_new, inplace=True)


# groups to be set to na or grouped to another
TYPE_CAR_CLEAN_MAP = {
    'num_bathroom_zillow': lambda x: x > 6,
    'num_bedroom': lambda x: x >= 7,
    'rank_building_quality': lambda x: x in (6, 8, 11, 12),
    'num_fullbath': lambda x: x >= 7,
    'num_garage': lambda x: x >= 5,
    'type_heating_system': lambda x: x in ('13', '20', '18', '11', '1', '14', '12', '10'),
    'type_landuse': lambda x: x in ('260', '263', '264', '267', '275', '31', '47'),
    'num_room': lambda x: x > 11 or (x < 3 and x != 0),
    'num_34_bathroom': lambda x: x >= 2,
    'num_unit': lambda x: x >= 5,
    'num_story': lambda x: x >= 4,
    'num_fireplace': lambda x: x >= 3
}


def create_type_var(data, col):
    """create type_var from given col.
        1, group or mark small categories as NA.
        2, also do this for num_ vars, and transform them to cat.
        3, create a new col groupby_col in data"""

    def clean_type_var(to_val):
        new_col_data = data[col].copy()
        new_col_data[new_col_data.index[new_col_data.apply(TYPE_CAR_CLEAN_MAP[col])]] = to_val
        data[new_col_name] = new_col_data
        cat_num_to_str(data, new_col_name)

    new_col_name = 'groupby__' + col

    if col in ('type_air_conditioning', 'flag_pool', 'flag_tax_delinquency',
               'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
               'raw_block', 'raw_census', 'block', 'census'):
        # already grouped in basic cleaning
        data[new_col_name] = data[col]

    if col == 'num_bathroom_zillow':
        clean_type_var(6)

    if col == 'num_bedroom':
        clean_type_var(7)

    if col == 'rank_building_quality':
        clean_type_var(np.nan)

    if col == 'code_fips':
        data[new_col_name] = data[col].copy()

    if col == 'num_fireplace':
        clean_type_var(3)

    if col == 'num_fullbath':
        clean_type_var(6)

    if col == 'num_garage':
        clean_type_var(4)

    if col == 'type_heating_system':
        clean_type_var(np.nan)

    if col == 'type_landuse':
        clean_type_var(np.nan)

    if col == 'num_room':
        clean_type_var(np.nan)

    if col == 'num_34_bathroom':
        clean_type_var(np.nan)

    if col == 'num_unit':
        clean_type_var(np.nan)

    if col == 'num_story':
        clean_type_var(np.nan)

    return new_col_name


def load_data_raw():
    # init load data
    prop_data = pd.read_csv('data/properties_2016.csv', header=0)
    error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
    error_data['sale_month'] = error_data['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month
    property_cleaning(prop_data)

    for col in prop_data.columns:
        if prop_data[col].dtype == np.float64:
            prop_data.loc[:, col] = prop_data[col].astype(np.float32)

    train_data = error_data.merge(prop_data, how='left', on='parcelid')
    # train_data.to_csv('data/train_data_merge.csv', index=False)

    submission = pd.read_csv('data/sample_submission.csv', header=0)
    submission['parcelid'] = submission['ParcelId']
    submission.drop('ParcelId', axis=1, inplace=True)
    test_data = submission.merge(prop_data, how='left', on='parcelid')

    clean_class3_var(train_data, test_data)

    return train_data, test_data


def clean_class3_var(train_data, test_data):

    # for those categories appear in tests only, fill with mode
    def clean_class3_var_inner(name):
        # fill test_only categories with mode
        train_col_nonan = train_data[name][~train_data[name].isnull()]
        test_col_nonan = test_data[name][~test_data[name].isnull()]
        train_cats = train_col_nonan.unique()
        test_cats = test_col_nonan.unique()
        test_only_cats = set(test_cats) - set(train_cats)
        marker = test_data[name].apply(lambda x: x in test_only_cats)
        test_data.loc[marker, name] = test_col_nonan.mode()[0]

    for name in ('code_city', 'code_neighborhood', 'code_zip', 'raw_block', 'block', 'str_zoning_desc', 'code_county_landuse'):
        clean_class3_var_inner(name)


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


def load_data_naive_lgb_v2(train_data, test_data):
    keep_feature = list(feature_info.index.values)
    for col in ['census_block', 'raw_census_block', 'year_assess']:
        keep_feature.remove(col)
    for col in ['num_bathroom_assessor', 'code_county', 'area_living_finished_calc', 'area_firstfloor_assessor']:
        keep_feature.remove(col)
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    return test_x, train_x, train_y


def prep_for_lgb(train_x, test_x):
    """map categorical variables to int for lgb run"""
    for col in set(train_x.columns).intersection(set(test_x.columns)):
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col(train_x, test_x, col)


def prep_for_lgb_single(data):
    """map categorical variables to int for lgb run"""
    for col in data.columns:
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col_single(data, col)


def load_data_naive_lgb(train_data, test_data):
    keep_feature = list(feature_info.index[feature_info['class'].apply(lambda x: True if x in {1, 2} else False)].values)
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

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

    return test_x, train_x, train_y


def load_data_naive_lgb_submit(train_data, test_data):
    """use subset of good features to see how it performance"""
    keep_feature = list(feature_info.index[feature_info['class'].apply(lambda x: True if x in {1, 2} else False)].values)
    keep_feature += [nmap_new_to_orig[n] for n in ['area_living_type_15',
                                                   'area_firstfloor_zillow',
                                                   'area_yard_patio',
                                                   'year_tax_due']]
    if len(keep_feature) != len(set(keep_feature)):
        raise Exception('duplicated feature')
    for f in ['code_fips', 'num_bathroom_assessor', 'area_living_type_12']:
        keep_feature.remove(f)
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    return test_x, train_x, train_y


def train_lgb_with_val(train_x, train_y, params_inp=params_naive):
    # train - validaiton split
    train_x_use, val_x, train_y_use, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

    # create lgb dataset
    lgb_train = lgb.Dataset(train_x_use, train_y_use)
    lgb_val = lgb.Dataset(val_x, val_y, reference=lgb_train)

    params = params_inp.copy()
    params.pop('num_boosting_rounds')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    valid_sets=lgb_val,
                    early_stopping_rounds=100)

    return gbm


def train_lgb(train_x, train_y, params_inp=params_naive):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = params_inp.copy()
    num_boost_round = params.pop('num_boosting_rounds')
    gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    return gbm


def train_xgb(train_x, train_y):
    pass


def cv_lgb_final(train_x, train_y, params_inp=params_naive):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = params_inp.copy()
    num_boost_round = params.pop('num_boosting_rounds')
    eval_hist = lgb.cv(params, lgb_train, num_boost_round=num_boost_round, early_stopping_rounds=30)
    return eval_hist['l1-mean'][-1]


def feature_importance(gbm, label=None, print_to_scr=True):
    """only applicable for niave gbm"""
    features = np.array(gbm.feature_name())
    n_features = features.shape[0]

    importance_split = np.array(gbm.feature_importance('split'))
    rank_split = np.argsort(np.argsort(importance_split))
    rank_split = n_features - rank_split
    importance_gain = np.array(gbm.feature_importance('gain'))
    rank_gain = np.argsort(np.argsort(importance_gain))
    rank_gain = n_features - rank_gain
    avg_rank = (rank_gain + rank_split) / 2
    rank_sort_idx = np.argsort(avg_rank)
    rank_sorted = avg_rank[rank_sort_idx]
    features_sorted = features[rank_sort_idx]

    importance_split = gbm.feature_importance('split')
    sort_split = np.argsort(importance_split)
    sort_split = sort_split[::-1]
    features_split_sort = features[sort_split]
    features_class_split = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_split_sort]
    importance_split_sort = importance_split[sort_split]

    importance_gain = gbm.feature_importance('gain')
    sort_gain = np.argsort(importance_gain)
    sort_gain = sort_gain[::-1]
    features_gain_sort = features[sort_gain]
    features_class_gain = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_gain_sort]
    importance_gain_sort = importance_gain[sort_gain]

    df_display = pd.DataFrame()
    df_display['features_avg'] = features_sorted
    df_display['avg_rank'] = rank_sorted
    df_display['class_avg'] = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_sorted]
    df_display['split_rank_avg'] = rank_split[rank_sort_idx]
    df_display['gain_rank_avg'] = rank_gain[rank_sort_idx]
    df_display['feature_split'] = features_split_sort
    df_display['class_split'] = features_class_split
    df_display['split'] = importance_split_sort
    df_display['feature_gain'] = features_gain_sort
    df_display['class_gain'] = features_class_gain
    df_display['gain'] = importance_gain_sort

    df_display = df_display[['features_avg', 'class_avg', 'avg_rank', 'split_rank_avg', 'gain_rank_avg',
                             'feature_split', 'class_split', 'split', 'feature_gain', 'class_gain', 'gain']]
    if label:
        df_display.to_csv('records/feature_importance_%s.csv' % label, index=False)
    if print_to_scr:
        print(df_display)
    return features_sorted


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
        rank_ref_split = list(feature_imp_naive_lgb['feature_split']).index(col_ref) + 1
        rank_ref_gain = list(feature_imp_naive_lgb['feature_gain']).index(col_ref) + 1
        rank_ref_avg = list(feature_imp_naive_lgb['feature_avg']).index(col_ref) + 1
        print('%s: rank split = %d, rank gain = %d, avg_rank = %.1f' % (col_ref, rank_ref_split, rank_ref_gain, rank_ref_avg))


# def search_lgb_bo(train_x, train_y, label='', n_iter=80):
#     lgb_train = lgb.Dataset(train_x, train_y)
#
#     def lgb_evaluate(num_leaves,
#                      min_data_in_leaf,
#                      learning_rate_log,
#                      lambda_l2_log):
#         learning_rate = 0.1 ** learning_rate_log
#         lambda_l2 = 0.1 ** lambda_l2_log
#         num_leaves = int(num_leaves)
#         min_data_in_leaf = int(min_data_in_leaf)
#         params = {
#             'boosting_type': 'gbdt',
#             'objective': 'regression_l1',
#             'metric': {'l1'},
#             'num_leaves': num_leaves,
#             'min_data_in_leaf': min_data_in_leaf,
#             'learning_rate': learning_rate,
#             'lambda_l2': lambda_l2,
#             'feature_fraction': 0.8,
#             'bagging_fraction': 0.7,
#             'bagging_freq': 5,
#             'verbosity': 0
#         }
#
#         eval_hist = lgb.cv(params, lgb_train, num_boost_round=3000, early_stopping_rounds=30)
#         return -eval_hist['l1-mean'][-1]
#
#     search_range = {'num_leaves': (30, 50),
#                     'min_data_in_leaf': (200, 500),
#                     'learning_rate_log': (1, 3),
#                     'lambda_l2_log': (1, 4)}
#     lgb_bo = BayesianOptimization(lgb_evaluate, search_range)
#     lgb_bo.maximize(n_iter=n_iter, init_points=5)
#
#     res = lgb_bo.res['all']
#     res_df = pd.DataFrame()
#     res_df['score'] = -np.array(res['values'])
#     for v in search_range:
#         if v in ('learning_rate_log', 'lambda_l2_log'):
#             v_label = v[:-4]
#             apply_func = lambda x: 0.1 ** x
#         else:
#             v_label = v
#             apply_func = lambda x: x
#         res_df[v_label] = np.array([apply_func(d[v]) for d in res['params']])
#     res_df.to_csv('temp_cv_res_bo_%s.csv' % label, index=False)
#     print('BO search finished')


def search_lgb_random(train_x, train_y, label='', n_iter=80):
    lgb_train = lgb.Dataset(train_x, train_y)

    def rand_min_data_in_leaf():
        return np.random.randint(200, 500)

    def rand_learning_rate():
        return np.random.uniform(1, 3)

    def rand_num_leaf():
        return np.random.randint(30, 50)

    def rand_lambda_l2():
        return np.random.uniform(1, 4)

    res = []
    for i in range(1, n_iter + 1):
        rand_params = {'num_leaves': rand_num_leaf(),
                       'min_data_in_leaf': rand_min_data_in_leaf(),
                       'learning_rate': 0.1 ** rand_learning_rate(),
                       'lambda_l2': 0.1 ** rand_lambda_l2()}
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
        eval_hist = lgb.cv(params, lgb_train, num_boost_round=3000, early_stopping_rounds=30)
        res.append([eval_hist['l1-mean'][-1],
                    rand_params['num_leaves'],
                    rand_params['min_data_in_leaf'],
                    rand_params['learning_rate'],
                    rand_params['lambda_l2']])
        print('finished %d / %d' % (i, n_iter))
    res_df = pd.DataFrame(res, columns=['score', 'num_leaves', 'min_data_in_leaf', 'learning_rate', 'lambda_l2'])
    res_df.to_csv('temp_cv_res_random_%s.csv' % label, index=False)


def search_lgb_grid(train_x, train_y):
    lgb_train = lgb.Dataset(train_x, train_y)

    min_data_in_leaf_list = [200, 230, 260]
    learning_rate_list = [0.005, 0.008, 0.01]
    num_leaf_list = [35, 40, 45]
    lambda_l2_list = [0.01, 0.02, 0.03]

    # min_data_in_leaf_list = [200]
    # learning_rate_list = [0.005]
    # num_leaf_list = [35]
    # lambda_l2_list = [0.01]

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
                    eval_hist = lgb.cv(params, lgb_train, num_boost_round=3000, early_stopping_rounds=30)
                    res.append([eval_hist['l1-mean'][-1],
                                num_leaf,
                                min_data_in_leaf,
                                learning_rate,
                                lambda_l2])
                    iter += 1
                    print('finished %d / %d' % (iter, n_trial))
    res_df = pd.DataFrame(res, columns=['score', 'num_leaves', 'min_data_in_leaf', 'learning_rate', 'lambda_l2'])
    res_df.to_csv('records/temp_cv_res_grid.csv', index=False)


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
    test_x, train_x, train_y = load_data_naive_lgb_v2(train_data, test_data)

    trained_gbm = train_lgb(train_x, train_y)
    pred_score = trained_gbm.predict(test_x)
    submit_nosea(pred_score, test_data['parcelid'], 2)


def pred_nosea_2step():
    train_data, test_data = load_data_raw()
    test_x, train_x, train_y = load_data_naive_lgb_v2(train_data, test_data)

    x = train_x['year_built'].fillna(1955)
    y = train_y.astype(np.float32)
    x = x.astype(np.float32)
    x = x.reshape(x.shape[0], 1)
    lr = LinearRegression()
    lr.fit(x, y)

    error2 = y - lr.predict(x)
    gbm = train_lgb(train_x, error2)

    test_year_built = test_x['year_built'].fillna(1955).reshape(test_x.shape[0], 1)
    pred_score = gbm.predict(test_x) + lr.predict(test_year_built)
    submit_nosea(pred_score, test_data['parcelid'], 1)


NUM_VARS = ['area_lot', 'area_living_type_12', 'dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
            'area_garage', 'area_pool']
#
CAT_VARS = ['type_air_conditioning', 'flag_pool', 'flag_tax_delinquency', 'type_heating_system', 'type_landuse',
            'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
            'raw_block', 'raw_census', 'block', 'census',
            'num_bathroom_zillow', 'num_bedroom', 'rank_building_quality', 'code_fips', 'num_fireplace', 'num_fullbath',
            'num_garage', 'num_room', 'num_unit', 'num_story']

# NUM_VARS = ['area_living_type_12']

# CAT_VARS = ['num_unit']


def new_feature_base_all(train_x_inp, test_x_inp):

    areas = ('area_lot', 'area_garage', 'area_pool', 'area_living_type_12', 'area_living_type_15')
    vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total')

    def feature_engineering_inner(target_data, raw_data):
        """target_data are those used for model input, it is output from naive lgb selection, thus does not contain all features"""

        # dollar_taxvalue variables
        target_data['dollar_taxvalue_structure_land_diff'] = raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']
        target_data['dollar_taxvalue_structure_land_absdiff'] = np.abs(raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land'])
        target_data['dollar_taxvalue_structure_land_diff_norm'] = (raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']) / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_structure_land_absdiff_norm'] = np.abs(raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']) / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_structure_total_ratio'] = raw_data['dollar_taxvalue_structure'] / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_structure_land_ratio'] = raw_data['dollar_taxvalue_structure'] / raw_data['dollar_taxvalue_land']
        target_data['dollar_taxvalue_total_structure_ratio'] = raw_data['dollar_taxvalue_total'] / raw_data['dollar_taxvalue_structure']
        target_data['dollar_taxvalue_total_land_ratio'] = raw_data['dollar_taxvalue_total'] / raw_data['dollar_taxvalue_land']
        target_data['dollar_taxvalue_land_structure_ratio'] = raw_data['dollar_taxvalue_land'] / raw_data['dollar_taxvalue_structure']

        # per_square variables
        for v in vars:
            for a in areas:
                target_data[v + '_per_' + a] = raw_data[v] / raw_data[a]
                target_data.loc[np.abs(raw_data[a]) < 1e-5, v + '_per_' + a] = np.nan

    test_x = test_x_inp.copy()
    train_x = train_x_inp.copy()
    feature_engineering_inner(test_x, test_x_inp)
    feature_engineering_inner(train_x, train_x_inp)

    return train_x, test_x


def new_feature_base_selected(train_x_inp, test_x_inp):

    areas = ('area_lot', 'area_garage', 'area_pool', 'area_living_type_12', 'area_living_type_15')
    vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total')

    def feature_engineering_inner(target_data, raw_data, run_type):
        """target_data are those used for model input, it is output from naive lgb selection, thus does not contain all features"""

        # dollar_taxvalue variables
        target_data['dollar_taxvalue_structure_land_diff'] = raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']
        target_data['dollar_taxvalue_structure_land_absdiff'] = np.abs(raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land'])
        target_data['dollar_taxvalue_structure_land_diff_norm'] = (raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']) / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_structure_land_absdiff_norm'] = np.abs(raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']) / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_structure_total_ratio'] = raw_data['dollar_taxvalue_structure'] / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_total_dollar_tax_ratio'] = raw_data['dollar_taxvalue_total'] / raw_data['dollar_tax']
        target_data['living_area_proportion'] = raw_data['area_living_type_12'] / raw_data['area_lot']

        if run_type == 'test':
            created_var_names = ['dollar_taxvalue_structure_land_diff', 'dollar_taxvalue_structure_land_absdiff',
                                 'dollar_taxvalue_structure_land_diff_norm', 'dollar_taxvalue_structure_land_absdiff_norm',
                                 'dollar_taxvalue_structure_total_ratio', 'dollar_taxvalue_total_dollar_tax_ratio', 'living_area_proportion']

        # per_square variables
        for v in vars:
            for a in areas:
                if (v == 'dollar_tax' and a in ('area_living_type_15', 'area_garage')) or \
                        (v == 'dollar_taxvalue_total' and a == 'area_living_type_15') or \
                        (a == 'area_pool'):
                    continue
                col_name = v + '_per_' + a
                if run_type == 'test':
                    created_var_names.append(col_name)
                target_data[col_name] = raw_data[v] / raw_data[a]
                target_data.loc[np.abs(raw_data[a]) < 1e-5, col_name] = np.nan

        return created_var_names if run_type == 'test' else None

    test_x = test_x_inp.copy()
    train_x = train_x_inp.copy()
    created_var_names = feature_engineering_inner(test_x, test_x_inp, 'test')
    feature_engineering_inner(train_x, train_x_inp, 'train')

    return train_x, test_x, created_var_names


def groupby_feature_gen(num_col, group_col, data, raw_data, test_data, op_type):
    """data: already applied lgb prep, can be directly used for lgb train.
       raw_data: type_var before lgb prep, used for group_var and mean_var creation.
       test_data: full data, used to determine nan index.
       created feature follows format num_var__groupby__cat_var__op_type"""
    new_cat_var = create_type_var(raw_data, group_col)
    _ = create_type_var(test_data, group_col)

    # first find too small groups and remove, too small groups are defined as
    # 1, first sort groups by group size.
    # 2, cumsum <= total non-nan samples * 0.001
    test_group_count = test_data[new_cat_var].groupby(test_data[new_cat_var]).count()
    test_group_count = test_group_count.sort_values()
    small_group_test = set(test_group_count.index[test_group_count.cumsum() < int(np.sum(~test_data[new_cat_var].isnull()) * 0.001)])

    data_group_count = raw_data[new_cat_var].groupby(raw_data[new_cat_var]).count()
    data_group_count = data_group_count.sort_values()
    small_group_data = set(data_group_count.index[data_group_count.cumsum() < int(np.sum(~raw_data[new_cat_var].isnull()) * 0.001)])

    small_group = small_group_data | small_group_test
    rm_idx = raw_data[new_cat_var].apply(lambda x: x in small_group)

    # now create variables
    # group_avg
    avg_col = raw_data[new_cat_var].map(raw_data[num_col].groupby(raw_data[new_cat_var]).mean())
    # neu col
    if op_type == 'neu':
        new_col = num_col + '__' + new_cat_var + '__neu'
        data[new_col] = (raw_data[num_col] - avg_col) / avg_col
        data.loc[avg_col < 1e-5, new_col] = np.nan
    elif op_type == 'absneu':
        new_col = num_col + '__' + new_cat_var + '__absneu'
        data[new_col] = np.abs(raw_data[num_col] - avg_col) / avg_col
        data.loc[avg_col < 1e-5, new_col] = np.nan
    else:
        raise Exception('unknown op type')

    # rm small group idx
    data.loc[rm_idx, new_col] = np.nan
    # clear newly created col
    test_data.drop(new_cat_var, axis=1, inplace=True)
    raw_data.drop(new_cat_var, axis=1, inplace=True)

    return new_col


def groupby_feature_gen_batch(num_vars, group_col, data, raw_data, test_data):
    """generate neu and group feature for all num features with given groupby feature"""
    new_features = []
    for num_var in num_vars:
        for op_type in ('neu', 'absneu'):
            new_features.append(groupby_feature_gen(num_var, group_col, data, raw_data, test_data, op_type))
    return new_features


def feature_engineering1(train_x_inp, test_x_inp, train_y):
    """input are ouptut from load_data_naive_lgb"""
    keep_num = 100  # keep 80 features at the end of each iteration
    # data preprocessing
    train_x_fe, test_x_fe, new_num_vars = new_feature_base_selected(train_x_inp, test_x_inp)  # copy is craeted here
    train_x_fe_raw = train_x_fe.copy()
    prep_for_lgb_single(train_x_fe)

    num_vars = NUM_VARS + new_num_vars
    # num_vars = NUM_VARS

    for cat_var in CAT_VARS:
        old_features = list(train_x_fe.columns)
        new_features = groupby_feature_gen_batch(num_vars, cat_var, train_x_fe, train_x_fe_raw, test_x_fe)
        gbm = train_lgb(train_x_fe, train_y)
        features_sorted = feature_importance(gbm, 'fe1_after_%s' % cat_var, print_to_scr=False)
        rm_features = filter_features(new_features, old_features, dict(zip(features_sorted, list(range(1, len(features_sorted) + 1)))), train_x_fe)
        features_sorted = [f for f in features_sorted if f not in rm_features]
        features_selected = features_sorted[:keep_num] if len(features_sorted) >= keep_num else features_sorted
        train_x_fe = train_x_fe[features_selected]

    dump_feature_list(train_x_fe.columns, 'fe1')

    return train_x_fe, test_x_fe


def feature_engineering2(train_x_inp, test_x_inp, train_y):
    keep_ratio = 0.8  # new features has to rank high in both global wise and local wise to be considered.
    # data preprocessing
    train_x_fe, test_x_fe, new_num_vars = new_feature_base_selected(train_x_inp, test_x_inp)  # copy is craeted here
    train_x_fe_raw = train_x_fe.copy()
    prep_for_lgb_single(train_x_fe)

    num_vars = NUM_VARS + new_num_vars
    # num_vars = NUM_VARS

    for cat_var in CAT_VARS:
        old_features = list(train_x_fe.columns)
        new_features = groupby_feature_gen_batch(num_vars, cat_var, train_x_fe, train_x_fe_raw, test_x_fe)
        gbm = train_lgb(train_x_fe, train_y)
        features_sorted = feature_importance(gbm, 'fe2_after_%s' % cat_var, print_to_scr=False)
        rm_features = filter_features(new_features, old_features, dict(zip(features_sorted, list(range(1, len(features_sorted) + 1)))), train_x_fe)
        features_sorted = [f for f in features_sorted if f not in rm_features]

        # filter new features
        top_features_all = features_sorted[:int(len(features_sorted) * keep_ratio)]
        sorted_features_new = [f for f in features_sorted if f in new_features]
        top_features_new = sorted_features_new[:int(len(sorted_features_new) * keep_ratio)]
        ex_features = [f for f in new_features if (f not in top_features_all or f not in top_features_new)]
        features_selected = [f for f in features_sorted if f not in ex_features]

        train_x_fe = train_x_fe[features_selected]

    dump_feature_list(train_x_fe.columns, 'fe2')

    return train_x_fe, test_x_fe


def param_search_fe(test_x_inp, train_x_inp, train_y, fe_label):
    """input from load_data_naive_lgb"""
    train_x_fe, test_x_fe, new_num_vars = new_feature_base_selected(train_x_inp, test_x_inp)  # copy is craeted here
    prep_for_lgb(train_x_fe, test_x_fe)

    features_list = load_feature_list(fe_label)
    raw_features = [f for f in train_x_fe.columns if f in features_list]
    train_x_fe = train_x_fe[raw_features]
    for f in features_list:
        num_col, _, group_col, op_type = f.split('__')
        train_x_fe[f] = groupby_feature_gen(num_col, group_col, train_x_fe, train_x_inp, test_x_inp, op_type)

    search_lgb_random(train_x_fe, train_y, fe_label)


def rank_corr(col1, col2, data):
    col1_s, col2_s = data[col1], data[col2]
    na_idx = np.logical_or(col1_s.isnull(), col2_s.isnull())
    use_data_col1 = col1_s[~na_idx]
    use_data_col2 = col2_s[~na_idx]
    corr_val = np.corrcoef(np.argsort(np.argsort(use_data_col1)), np.argsort(np.argsort(use_data_col2)))[0,1]
    return 0.0 if np.isnan(corr_val) else corr_val


def rank_corr_matrix(features, data):
    n_f = len(features)
    df = pd.DataFrame(np.zeros((n_f, n_f)), columns=features, index=features)
    calced_features = []
    for idx1 in range(n_f):
        for idx2 in range(n_f):
            f1, f2 = features[idx1], features[idx2]
            calced_features.append((f1, f2))
            df.loc[f1, f2] = rank_corr(f1, f2, data)
    df.to_csv('rank_corr.csv')


def filter_features(features_new, features_old, features_rank, data):
    """pairwise filter features, for each pair (only calc corr within new features and between new features and old features), if 
        1, both-nonnan index rank corr is lower than 0.9, keep both.
        2, if higher, check count of xor nan index, if more than 10 precent of n_sample, keep both.
        3, else keep one that has higher rank.
       input features_rank as a dictionary for quick reference.
       return list of to-be-removed new features"""

    def select_feature(f1, f2):
        # NOTE, use abs corr here, as tree is indifferent to order
        if np.abs(rank_corr(f1, f2, data)) > 0.9:
            nan_diff = np.sum(np.logical_xor(data[f1].isnull(), data[f2].isnull()))
            if nan_diff < data.shape[0] * 0.1:
                feature_list = [f1, f2]
                ranks = [features_rank[f] for f in feature_list]
                return feature_list[np.argsort(ranks)[0]]
        return None

    rm_features = set()
    n_new_features = len(features_new)
    for idx1 in range(n_new_features):
        for idx2 in range(idx1 + 1, n_new_features):
            rm_f = select_feature(features_new[idx1], features_new[idx2])
            if rm_f:
                rm_features.add(rm_f)

    for new_f in features_new:
        for old_f in features_old:
            rm_f = select_feature(new_f, old_f)
            if rm_f:
                rm_features.add(rm_f)

    return rm_features


def seasonality_analysis(train_x, train_y, month_col):
    n_iter = 100
    stack_months = []
    stack_mean_erros = []
    for idx in range(n_iter):
        x, x_cv, y, y_cv = train_test_split(train_x, train_y, test_size=0.33, random_state=idx*3+1, stratify=month_col)
        gbm = train_lgb(x, y)
        y_pred = gbm.predict(x_cv)
        error = y_cv - y_pred
        error_by_month = pd.Series(error).groupby(month_col).mean()
        stack_months += list(error_by_month.index.values)
        stack_mean_erros += list(error_by_month.values)
    sns.barplot(x=stack_months, y=stack_mean_erros).set_title('mean_error distribution against month')


def dump_feature_list(features, label):
    txtf = open('feature_list_%s.txt' % label, 'w')
    s = ''
    for f in features:
        s += f + '\n'
    txtf.write(s)
    txtf.close()


def load_feature_list(label):
    txtf = open('feature_list_%s.txt' % label, 'r')
    features_list_raw = txtf.read()
    txtf.close()

    features = []
    for f in features_list_raw.split('\n'):
        if f:
            features.append(f)
    return features







