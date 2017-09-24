import numpy as np
import pandas as pd
import math
import gc
from time import time
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
# from bayes_opt import BayesianOptimization
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')
nmap_orig_to_new =  dict(zip(feature_info['orig_name'].values, feature_info.index.values))
nmap_new_to_orig =  dict(zip(feature_info.index.values, feature_info['orig_name'].values))

feature_imp_naive_lgb = pd.read_csv('records/feature_importance_raw_all.csv')

params_naive = {
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
    'verbosity': 0,
    'num_boosting_rounds': 1500
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


def cv_meta_model(x, y, train_model_func, outlier_thresh_y=0.001, outlier_handling=None, seanality_handling=None):
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
            train_x_cv, train_y_cv = x.iloc[idx[-fold_size:], :].copy(), y.iloc[idx[-fold_size:]].copy()
            test_x_cv, test_y_cv = x.iloc[idx[:-fold_size], :].copy(), y.iloc[idx[:-fold_size]].copy()
        else:
            train_x_cv = pd.concat([x.iloc[idx[: fold_size * n], :].copy(), x.iloc[idx[fold_size * (n + 1):], :].copy()])
            train_y_cv = pd.concat([y.iloc[idx[: fold_size * n]].copy(), y.iloc[idx[fold_size * (n + 1):]].copy()])
            test_x_cv = x.iloc[idx[fold_size * n: fold_size * (n + 1)], :].copy()
            test_y_cv = y.iloc[idx[fold_size * n: fold_size * (n + 1)]].copy()

        if outlier_handling:
            train_x_cv, train_y_cv = outlier_handling(train_x_cv, train_y_cv, outlier_thresh_y)
        model = train_model_func(train_x_cv, train_y_cv)
        pred_y_cv = model.predict(test_x_cv)
        evals[n] = np.mean(np.abs(pred_y_cv - test_y_cv))
    return np.mean(evals)


def outlier_y_rm(train_x, train_y, thresh):
    quantile_cut = thresh
    down_thresh, up_thresh = train_y.quantile([quantile_cut, 1 - quantile_cut])
    pos_ex_y_idx = train_y > up_thresh
    neg_ex_y_idx = train_y < down_thresh
    ex_y_idx = np.logical_or(pos_ex_y_idx, neg_ex_y_idx)
    # remove outlier
    train_x = train_x.loc[~ex_y_idx, :]
    train_y = train_y[~ex_y_idx]
    return train_x, train_y


def outlier_y_capfloor(train_x, train_y, thresh):
    quantile_cut = thresh
    down_thresh, up_thresh = train_y.quantile([quantile_cut, 1 - quantile_cut])
    pos_ex_y_idx = train_y > up_thresh
    neg_ex_y_idx = train_y < down_thresh
    # cap_floor outlier
    train_y[pos_ex_y_idx] = up_thresh
    train_y[neg_ex_y_idx] = down_thresh
    return train_x, train_y


def outlier_x_clean(train_x, train_y, test_x, type_train='rm', type_test='na'):
    """2 strategies for x in train: remove, set to NA. (cap / floor is not expected to help for a tree based model).
       2 strategies for test: set to NA, leave as it is. 
       For each numerical variables, need to consider to do one-side or 2-sided cleaning"""
    rm_idx_train = []

    def proc_outlier_num(col, q_down, q_up):
        """use -1 if one side of data is not trimmed"""
        name = nmap_new_to_orig[col] if col in nmap_new_to_orig else col
        thresh_up = test_x[name].quantile(q_up) if q_up > 0 else np.inf  # Series.quantile has already considered NA
        thresh_down = test_x[name].quantile(q_down) if q_down > 0 else -np.inf

        pos_idx_train = train_x[name] >= thresh_up
        neg_idx_train = train_x[name] <= thresh_down
        pos_idx_test = test_x[name] >= thresh_up
        neg_idx_test = test_x[name] <= thresh_down
        idx_train = np.logical_or(pos_idx_train, neg_idx_train)
        idx_test = np.logical_or(pos_idx_test, neg_idx_test)
        rm_idx_train.append(idx_train)
        train_x.loc[idx_train, name] = np.nan
        if type_test == 'na':
            test_x.loc[idx_test, name] = np.nan

    # year built
    proc_outlier_num('year_built', 0.001, -1)

    # latitude

    # num_bathroom_zillow
    idx_train = train_x['calculatedbathnbr'] > 6
    rm_idx_train.append(idx_train)
    train_x.loc[train_x.index[idx_train], 'calculatedbathnbr'] = np.nan
    if type_test == 'na':
        idx_test = test_x['calculatedbathnbr'] > 6
        test_x.loc[test_x.index[idx_test], 'calculatedbathnbr'] = np.nan


def outlier_rm_x(train_x, train_y):
    rm_idx = []
    return train_x, train_y


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
    'num_room': lambda x: x > 11 or x < 3,
    'num_34_bathroom': lambda x: x >= 2,
    'num_unit': lambda x: x >= 5,
    'num_story': lambda x: x >= 4
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

    new_col_name = 'groupby_' + col

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
    # for col in ['num_bathroom_assessor', 'code_county', 'area_living_type_12', 'area_firstfloor_assessor']:
    #     keep_feature.remove(col)
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


def cv_lgb_final(train_x, train_y, params_inp=params_naive):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = params_inp.copy()
    num_boost_round = params.pop('num_boosting_rounds')
    eval_hist = lgb.cv(params, lgb_train, num_boost_round=num_boost_round)
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


# def search_lgb_bo(train_x, train_y):
#     n_iter = 30
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
#         eval_hist = lgb.cv(params, lgb_train, num_boost_round=2500, early_stopping_rounds=30)
#         return -eval_hist['l1-mean'][-1]
#
#     search_range = {'num_leaves': (30, 50),
#                     'min_data_in_leaf': (200, 500),
#                     'learning_rate_log': (1, 3),
#                     'lambda_l2_log': (1, 3)}
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
#     res_df.to_csv('temp_cv_res_bo.csv', index=False)
#     print('BO search finished')


def search_lgb_random(train_x, train_y):
    lgb_train = lgb.Dataset(train_x, train_y)

    def rand_min_data_in_leaf():
        return np.random.randint(200, 500)

    def rand_learning_rate():
        return np.random.uniform(1, 4)

    def rand_num_leaf():
        return np.random.randint(30, 50)

    def rand_lambda_l2():
        return np.random.uniform(1, 4)

    n_trail = 30
    res = []
    for i in range(1, n_trail + 1):
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
        eval_hist = lgb.cv(params, lgb_train, num_boost_round=2500, early_stopping_rounds=30)
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
    test_x, train_x, train_y = load_data_naive_lgb(train_data, test_data)
    trained_gbm = train_lgb(train_x, train_y)
    pred_score = trained_gbm.predict(test_x)
    submit_nosea(pred_score, test_data['parcelid'], 2)


NUM_VARS = ['area_lot', 'area_living_finished_calc', 'dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
            'area_garage', 'area_pool']

CAT_VARS = ['type_air_conditioning', 'flag_pool', 'flag_tax_delinquency',
            'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
            'raw_block', 'raw_census', 'block', 'census',
            'num_bathroom_zillow', 'num_bedroom', 'rank_building_quality', 'code_fips', 'num_fireplace', 'num_fullbath',
            'num_garage', 'type_heating_system', 'type_landuse', 'num_room', 'num_34_bathroom', 'num_unit', 'num_story']


def new_feature_base_all(train_x_inp, test_x_inp):

    areas = ('area_lot', 'area_garage', 'area_pool', 'area_living_finished_calc', 'area_living_type_15')
    vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total')

    def feature_engineering_inner(target_data, raw_data):
        """target_data are those used for model input, it is output from naive lgb selection, thus does not contain all features"""

        # dollar_taxvalue variables
        target_data['dollar_taxvalue_structure_land_diff'] = raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']
        target_data['dollar_taxvalue_structure_land_absdiff'] = np.abs(raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land'])
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

    areas = ('area_lot', 'area_garage', 'area_pool', 'area_living_finished_calc', 'area_living_type_15')
    vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total')

    created_var_names = []

    def feature_engineering_inner(target_data, raw_data, run_type):
        """target_data are those used for model input, it is output from naive lgb selection, thus does not contain all features"""

        # dollar_taxvalue variables
        target_data['dollar_taxvalue_structure_land_diff'] = raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land']
        target_data['dollar_taxvalue_structure_land_absdiff'] = np.abs(raw_data['dollar_taxvalue_structure'] - raw_data['dollar_taxvalue_land'])
        target_data['dollar_taxvalue_structure_total_ratio'] = raw_data['dollar_taxvalue_structure'] / raw_data['dollar_taxvalue_total']
        target_data['dollar_taxvalue_total_structure_ratio'] = raw_data['dollar_taxvalue_total'] / raw_data['dollar_taxvalue_structure']

        if run_type == 'test':
            created_var_names = ['dollar_taxvalue_structure_land_diff', 'dollar_taxvalue_structure_land_absdiff',
                                 'dollar_taxvalue_structure_total_ratio', 'dollar_taxvalue_total_structure_ratio']

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

    test_x = test_x_inp.copy()
    train_x = train_x_inp.copy()
    feature_engineering_inner(test_x, test_x_inp, 'test')
    feature_engineering_inner(train_x, train_x_inp, 'train')

    return train_x, test_x, created_var_names


def group_feature_gen(data_train, raw_data_train, raw_data_test, cat_var, num_vars):
    """generate diff_mean & ratio_mean across cat_var groups for all numerical variables.
       input raw_data should be output from new_feature_base_selected(), i.e containing all original information of cat & numerical vars.
       input data should be the output from last feature engineering iteration.
       data and raw_data are expected to be of same length.
       Also create the feature for test data and save to local, so that it is easier to be used in the future"""
    new_cat_var = create_type_var(raw_data_train, cat_var)
    _ = create_type_var(raw_data_test, cat_var)

    # region parcel count
    first_num_var = num_vars[0]
    raw_data_train = raw_data_train.join(raw_data_train[first_num_var].groupby(raw_data_train[new_cat_var]).count(), on=new_cat_var, rsuffix='_%s_count' % new_cat_var)
    raw_data_test = raw_data_test.join(raw_data_test[first_num_var].groupby(raw_data_test[new_cat_var]).count(), on=new_cat_var, rsuffix='_%s_count' % new_cat_var)
    count_col = first_num_var + '_%s_count' % new_cat_var
    count_thresh = raw_data_test[count_col].quantile(0.001)
    set_na_idx_train = raw_data_train[count_col] <= count_thresh
    set_na_idx_test = raw_data_test[count_col] <= count_thresh
    raw_data_train.drop(count_col, axis=1, inplace=True)
    raw_data_test.drop(count_col, axis=1, inplace=True)
    for num_var in num_vars:
        # group avg
        raw_data_train = raw_data_train.join(raw_data_train[num_var].groupby(new_cat_var).mean(), on=new_cat_var, rsuffix='_%s_avg' % new_cat_var)
        raw_data_test = raw_data_test.join(raw_data_test[num_var].groupby(new_cat_var).mean(), on=new_cat_var, rsuffix ='_%s_avg' % new_cat_var)
        # neutral
        data_train[num_var + '_%s_neu' % new_cat_var] = raw_data_train[num_var] - raw_data_train[num_var + '_%s_avg' % new_cat_var]

        neu_col_test = raw_data_test[num_var] - raw_data_test[num_var + '_%s_avg' % new_cat_var]
        neu_col_test[set_na_idx_test] = np.nan
        neu_col_test.to_csv('fe/' + num_var + '_%s_neu' % new_cat_var + '.csv')
        # ratio
        data_train[num_var + '_%s_ratio' % new_cat_var] = raw_data_train[num_var] / raw_data_train[num_var + '_%s_avg' % new_cat_var]
        data_train.loc[data_train.index[raw_data_train[num_var + '_%s_avg' % new_cat_var] < 1e-5], num_var + '_%s_ratio' % new_cat_var] = np.nan

        ratio_col_test = raw_data_test[num_var] / raw_data_test[num_var + '_%s_avg' % new_cat_var]
        ratio_col_test[raw_data_test[num_var + '_%s_avg' % new_cat_var] < 1e-5] = np.nan
        ratio_col_test[set_na_idx_test] = np.nan
        ratio_col_test.to_csv('fe/' + num_var + '_%s_ratio' % new_cat_var + '.csv')

        raw_data_train.drop(num_var + '_%s_avg' % new_cat_var, axis=1, inplace=True)
        raw_data_test.drop(num_var + '_%s_avg' % new_cat_var, axis=1, inplace=True)
        # set small groups to nan
        data_train.loc[set_na_idx_train, [num_var + '_%s_neu' % new_cat_var, num_var + '_%s_ratio' % new_cat_var]] = np.nan

    raw_data_train.drop(new_cat_var, axis=1, inplace=True)
    raw_data_test.drop(new_cat_var, axis=1, inplace=True)


def feature_engineering(train_x_inp, test_x_inp, train_y):
    keep_num = 80
    train_x_fe, test_x_fe, new_num_vars = new_feature_base_selected(train_x_inp, test_x_inp)
    train_x_fe_base = train_x_fe.copy()

    num_vars = NUM_VARS + new_num_vars

    for cat_var in CAT_VARS:
        group_feature_gen(train_x_fe, train_x_fe_base, test_x_fe, cat_var, num_vars)
        gbm = train_lgb(train_x_fe, train_y)
        features_sorted = feature_importance(gbm, 'after_%s' % cat_var, False)
        features_selected = features_sorted[:keep_num] if features_sorted.shape[0] >= keep_num else features_sorted
        train_x_fe = train_x_fe[features_selected]

    return train_x_fe, test_x_fe

# CANNOT use parcelid to join for train data, it is not unique









