import pandas as pd
import numpy as np
import gc

# python version 3.5.3


def make_submission(col_2016, col_2017, label):
    submission = pd.read_csv('data/sample_submission.csv', header=0)
    df = pd.DataFrame()
    df['ParcelId'] = submission['ParcelId']

    df['201610'] = col_2016
    df['201611'] = col_2016
    df['201612'] = col_2016
    df['201710'] = col_2017
    df['201711'] = col_2017
    df['201712'] = col_2017

    df.to_csv('data/submission_%s.csv.gz' % label, index=False, float_format='%.7f', compression='gzip')


# to run below script, need to set up a /data folder with the code, and put the feature_info.csv file there.
# submission csv files will also be output under this folder.


# ------------------------------------------------------------ 2-step lightGBM prediction -----------------------------------------------------------------
# the 2-step idea is as follows:
# direct lightGBM fit is not ideal, even including months as a categorical feature in model does not make in-sample fit un-biased.
# i.e. within each month, the predicted errors have a non-zero median.
# if direct lightGBM fit cannot handle this, I added a second step fit, that is fitting the step-1 errors of target months (10, 11, 12) separately,
# well, with a shrink in sample-size, I fit it with only selected features.
# this second step is conducted with mainly the aim to adjust median, but hoping to capture some further local structure.
# consider it as a type of model stacking, just the stacked one is not same data trained with a different model family, but the same model family trained with different data.

import lightgbm as lgb  # version 2.0.5

# --------------------------------------------------------- parameters ------------------------------------------------------------
# params from random-search, num-boosting has been boosted considering sample size
raw_lgb_2y_1 = {'num_leaves': 53,
                'min_data_in_leaf': 366,
                'learning_rate': 0.002382,
                'num_boosting_rounds': 10500}

raw_lgb_2y_2 = {'num_leaves': 64,
                'min_data_in_leaf': 467,
                'learning_rate': 0.002088,
                'num_boosting_rounds': 11000}

raw_lgb_2y_3 = {'num_leaves': 61,
                'min_data_in_leaf': 344,
                'learning_rate': 0.003809,
                'num_boosting_rounds': 6655}

raw_lgb_2y_4 = {'num_leaves': 45,
                'min_data_in_leaf': 297,
                'learning_rate': 0.002589,
                'num_boosting_rounds': 11500}

raw_lgb_2y_5 = {'num_leaves': 73,
                'min_data_in_leaf': 264,
                'learning_rate': 0.0082,
                'num_boosting_rounds': 2400}


lgb_month_1 = {'num_leaves': 7,
             'min_data_in_leaf': 37,
             'learning_rate': 0.004607,
             'num_boosting_rounds': 360}


lgb_month_2 = {'num_leaves': 7,
             'min_data_in_leaf': 37,
             'learning_rate': 0.004607,
             'num_boosting_rounds': 480}


lgb_month_3 = {'num_leaves': 6,
                'min_data_in_leaf': 82,
                'learning_rate': 0.003154,
                'num_boosting_rounds': 600}


lgb_month_4 = {'num_leaves': 6,
                'min_data_in_leaf': 82,
                'learning_rate': 0.003154,
                'num_boosting_rounds': 800}

raw_lgb_2y = [raw_lgb_2y_1, raw_lgb_2y_2, raw_lgb_2y_3, raw_lgb_2y_4, raw_lgb_2y_5]
lgb_month = [lgb_month_1, lgb_month_2, lgb_month_3, lgb_month_4]


# replace high-cardinal categorical features with group-related numerical features
class3_new_features = ('dollar_taxvalue_structure_land_absdiff_norm__groupby__code_zip__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_neighborhood__mean',
                       'dollar_taxvalue_structure__groupby__code_city__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__raw_census__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__str_zoning_desc__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__code_county_landuse__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__census__mean',
                       'dollar_taxvalue_structure__groupby__block__mean')

class3_rm_features = ('code_zip_lgb',
                      'code_neighborhood_lgb',
                      'code_city_lgb',
                      'raw_census_lgb',
                      'raw_block_lgb',
                      'str_zoning_desc_lgb',
                      'code_county_landuse_lgb',
                      'census_lgb',
                      'block_lgb')

step2_keep_only_feature = ('year_built', 'area_lot', 'dollar_tax', 'area_living_type_12', 'dollar_taxvalue_structure', 'latitude', 'longitude', 'dollar_taxvalue_land',
                           'dollar_taxvalue_structure', 'dollar_taxvalue_total', 'area_garage') + class3_new_features

# this requires a hand-made file feature_info.csv, which uses a more meaningful name for columns and contains labeling of features whether it is numerical or categorical
# I need to keep it here for minimum change of code
feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')
nmap_orig_to_new = dict(zip(feature_info['orig_name'].values, feature_info.index.values))
nmap_new_to_orig = dict(zip(feature_info.index.values, feature_info['orig_name'].values))


# --------------------------------------------------------- data prep ------------------------------------------------------------
def cat_num_to_str(data, col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not data[col_name].dtype == 'O':
        data.loc[:, col_name] = data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def property_trans(prop_data):
    """only does transformation, no cleaning"""

    def mark_flag_col(col_name):
        """mark bool for numerical columns, mark True for val > 0, and False otherwise (include NaN)"""
        data_marks_true = prop_data[col_name] >= 0.5
        prop_data.loc[prop_data.index[data_marks_true], col_name] = 'TRUE'
        prop_data.loc[prop_data.index[~data_marks_true], col_name] = 'FALSE'

    cat_num_to_str(prop_data, 'airconditioningtypeid')
    cat_num_to_str(prop_data, 'architecturalstyletypeid')
    cat_num_to_str(prop_data, 'buildingclasstypeid')
    cat_num_to_str(prop_data, 'decktypeid')
    cat_num_to_str(prop_data, 'fips')
    mark_flag_col('hashottuborspa')
    cat_num_to_str(prop_data, 'heatingorsystemtypeid')
    mark_flag_col('poolcnt')
    mark_flag_col('pooltypeid10')
    mark_flag_col('pooltypeid2')
    mark_flag_col('pooltypeid7')
    prop_data['type_pool'] = 'None'
    prop_data.loc[prop_data.index[prop_data['pooltypeid2'] == 'TRUE'], 'type_pool'] = 'TRUE'
    prop_data.loc[prop_data.index[prop_data['pooltypeid7'] == 'TRUE'], 'type_pool'] = 'FALSE'
    cat_num_to_str(prop_data, 'propertycountylandusecode')
    cat_num_to_str(prop_data, 'propertylandusetypeid')
    cat_num_to_str(prop_data, 'propertyzoningdesc')

    # raw_census_block, raw_census, raw_block.
    # this part did wrong, without casting x * 1000000 to int, this use a scientific representation of float, which is meaningless converted to string
    prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
    prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('temp', axis=1, inplace=True)
    prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

    cat_num_to_str(prop_data, 'regionidcity')
    cat_num_to_str(prop_data, 'regionidcounty')
    cat_num_to_str(prop_data, 'regionidneighborhood')
    cat_num_to_str(prop_data, 'regionidzip')
    cat_num_to_str(prop_data, 'storytypeid')
    cat_num_to_str(prop_data, 'typeconstructiontypeid')
    mark_flag_col('fireplaceflag')

    # census_block
    prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
    prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('censustractandblock', axis=1, inplace=True)

    # name columns
    prop_data.rename(columns=nmap_orig_to_new, inplace=True)


def property_cleaning_v2(prop_data):
    """clean cols according to train_data distribution"""

    def clear_cat_col_group(col, groups):
        """set given groups of a categorical col to na"""
        in_group_flag = prop_data[col].apply(lambda x: x in groups)
        prop_data.loc[prop_data[col].index[in_group_flag], col] = np.nan

    # these are categories that exists in testing but not in training
    clear_cat_col_group('type_air_conditioning', ['12'])
    clear_cat_col_group('type_architectural_style', ['27', '5'])
    clear_cat_col_group('type_heating_system', ['19', '21'])
    clear_cat_col_group('type_construction', ['11'])
    clear_cat_col_group('type_landuse', ['270', '279'])

    # num_garage and area_garage not consistent
    mark_idx = prop_data.index[np.logical_and(np.abs(prop_data['area_garage'] - 0) < 1e-12, prop_data['num_garage'] > 0)]
    sub_df = prop_data.loc[mark_idx, ['area_garage', 'num_garage']]
    sub_df = sub_df.join(prop_data['area_garage'].groupby(prop_data['num_garage']).mean(), on='num_garage', rsuffix='_avg')
    prop_data.loc[mark_idx, 'area_garage'] = sub_df['area_garage_avg']


def prep_for_lgb_single(data):
    """map categorical variables to int for lgb run"""
    for col in data.columns:
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'cat':
            # convert to lgb usable categorical
            # set nan to string so that can be sorted, make sure training set and testing set get the same coding
            data.loc[data.index[data[col].isnull()], col] = 'nan'
            uni_vals = np.sort(data[col].unique()).tolist()
            map = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
            col_new = col + '_lgb'
            data[col_new] = data[col].apply(lambda x: map[x])
            data[col_new] = data[col_new].astype('category')


def create_type_var(data, col):
    """create type_var from given col.
        1, group or mark small categories as NA.
        2, also do this for num_ vars, and transform them to cat.
        3, create a new col groupby_col in data"""

    new_col_name = 'groupby__' + col

    if col in ('type_air_conditioning', 'flag_pool', 'flag_tax_delinquency',
               'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
               'raw_block', 'raw_census', 'block', 'census', 'code_fips'):
        # already grouped in basic cleaning
        data[new_col_name] = data[col].copy()
    else:
        data[new_col_name] = data[col].copy()
        cat_num_to_str(data, new_col_name)

    return new_col_name


def groupby_feature_gen_single(num_col, group_col, data, raw_data, op_type):
    """data: already applied lgb prep, can be directly used for lgb train.
       raw_data: type_var before lgb prep, used for group_var and mean_var creation.
       test_data: full data, used to determine nan index.
       created feature follows format num_var__groupby__cat_var__op_type"""
    new_cat_var = create_type_var(raw_data, group_col)

    # first find too small groups and remove, too small groups are defined as
    # 1, first sort groups by group size.
    # 2, cumsum <= total non-nan samples * 0.001
    group_count = raw_data[new_cat_var].groupby(raw_data[new_cat_var]).count()
    group_count = group_count.sort_values()
    small_group = set(group_count.index[group_count.cumsum() < int(np.sum(~raw_data[new_cat_var].isnull()) * 0.0001)])

    rm_idx = raw_data[new_cat_var].apply(lambda x: x in small_group)

    # now create variables
    # group_avg
    mean_col_name = num_col + '__' + new_cat_var + '__mean'
    if mean_col_name in raw_data.columns:
        avg_col = raw_data[mean_col_name]
    else:
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
    elif op_type == 'mean':
        new_col = num_col + '__' + new_cat_var + '__mean'
        data[new_col] = avg_col
    elif op_type == 'count':
        new_col = num_col + '__' + new_cat_var + '__count'
        data[new_col] = raw_data[new_cat_var].groupby(raw_data[new_cat_var]).count()
    else:
        raise Exception('unknown op type')

    # rm small group idx
    data.loc[rm_idx, new_col] = np.nan
    # clear newly created col
    raw_data.drop(new_cat_var, axis=1, inplace=True)

    return new_col


def feature_factory(col, data, keep_num_var=False):
    """add col to data, generated from raw data columns"""

    if col == 'dollar_taxvalue_structure_land_diff':
        data[col] = data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land']

    if col == 'dollar_taxvalue_structure_land_absdiff':
        data[col] = np.abs(data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land'])

    if col == 'dollar_taxvalue_structure_land_diff_norm':
        data[col] = (data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land']) / data['dollar_taxvalue_total']

    if col == 'dollar_taxvalue_structure_land_absdiff_norm':
        data[col] = np.abs(data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land']) / data['dollar_taxvalue_total']

    if col == 'dollar_taxvalue_structure_total_ratio':
        data[col] = data['dollar_taxvalue_structure'] / data['dollar_taxvalue_total']

    if col == 'dollar_taxvalue_total_dollar_tax_ratio':
        data[col] = data['dollar_taxvalue_total'] / data['dollar_tax']

    if col == 'living_area_proportion':
        data[col] = data['area_living_type_12'] / data['area_lot']

    if '__per__' in col:
        if not '__groupby__' in col:
            v, a = col.split('__per__')
            data[col] = data[v] / data[a]
            data.loc[np.abs(data[a]) < 1e-5, col] = np.nan

    if '__groupby__' in col:
        num_var, cat_var_and_op_type = col.split('__groupby__')
        cat_var, op_type = cat_var_and_op_type.split('__')
        num_var_in_raw_data = num_var in data.columns
        if not num_var_in_raw_data:
            feature_factory(num_var, data)
        _ = groupby_feature_gen_single(num_var, cat_var, data, data, op_type)
        if not num_var_in_raw_data:
            if not keep_num_var:
                data.drop(num_var, axis=1, inplace=True)
            else:
                print('num var kept: %s' % num_var)


def load_prop_data():
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)
    sub_index = sample_submission['ParcelId']
    prop_2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    prop_2016.index = prop_2016['parcelid']
    prop_2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)
    prop_2017.index = prop_2017['parcelid']

    def size_control(data):
        data.loc[:, ['latitude', 'longitude']] = data.loc[:, ['latitude', 'longitude']] / 1e6
        for col in data.columns:
            if data[col].dtype == np.float64:
                # this is a bad transformation here, it damages information of raw_census_block
                data.loc[:, col] = data[col].astype(np.float32)

    size_control(prop_2016)
    size_control(prop_2017)

    # make sure prediction index order is inline with expected submission index order
    prop_2016 = prop_2016.loc[sub_index, :]
    prop_2017 = prop_2017.loc[sub_index, :]

    property_trans(prop_2016)
    property_trans(prop_2017)

    property_cleaning_v2(prop_2016)
    property_cleaning_v2(prop_2017)

    n_row_prop_2016 = prop_2016.shape[0]
    n_row_prop_2017 = prop_2017.shape[0]
    prop_join = pd.concat([prop_2016, prop_2017], axis=0)

    prep_for_lgb_single(prop_join)
    prop_2016 = prop_join.iloc[list(range(n_row_prop_2016)), :]
    prop_2017 = prop_join.iloc[list(range(n_row_prop_2016, n_row_prop_2016 + n_row_prop_2017)), :]

    for f in class3_new_features:
        feature_factory(f, prop_2016)
        feature_factory(f, prop_2017)

    return prop_2016, prop_2017


def load_train_data(prop_2016, prop_2017):
    """if engineered features exists, it should be performed at prop_data level, and then join to error data"""

    train_2016 = pd.read_csv('data/train_2016_v2.csv')
    train_2017 = pd.read_csv('data/train_2017.csv')

    train_2016 = pd.merge(train_2016, prop_2016, how='left', on='parcelid')
    train_2016['data_year'] = 2016
    train_2017 = pd.merge(train_2017, prop_2017, how='left', on='parcelid')
    train_2017['data_year'] = 2017

    train = pd.concat([train_2016, train_2017], axis=0)
    train.index = list(range(train.shape[0]))
    train['sale_month'] = train['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month
    train_y = train['logerror']
    train_x = train.drop('logerror', axis=1)
    return train_x, train_y


# --------------------------------------------------------- model train and predict ------------------------------------------------------------
def rm_outlier(x, y):
    idx = np.logical_and(y < 0.7566, y > -0.4865)  # outlier bounds are determined by quantiles, and then hard coded here
    x = x.loc[x.index[idx], :]
    y = y[idx]
    return x, y


def lgb_data_prep(data, new_features=tuple(), rm_features=tuple(), keep_only_feature=()):
    keep_feature = list(feature_info.index.values)
    feature_info_copy = keep_feature.copy()

    for col in feature_info_copy:
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'cat' and col in keep_feature:
            keep_feature.remove(col)
            keep_feature.append(col + '_lgb')
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'none' and col in keep_feature:
            keep_feature.remove(col)

    for f in new_features:
        if f not in keep_feature:
            keep_feature.append(f)

    for col in rm_features:
         if col in keep_feature:
             keep_feature.remove(col)

    for col in ['num_bathroom_assessor', 'code_county', 'area_living_finished_calc', 'area_firstfloor_assessor']:
        # these columns are removed as the information contained is duplicated
        if col in keep_feature:
            keep_feature.remove(col)

    if len(keep_only_feature) > 0:
        keep_feature = list(keep_only_feature)

    return data[keep_feature]


def train_lgb(train_x, train_y, params_inp):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = params_inp.copy()
    num_boost_round = params.pop('num_boosting_rounds')
    gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    return gbm

params_base = {
    'boosting_type': 'gbdt',
    'feature_fraction': 0.95,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbosity': 0,
    'lambda_l2': 0,
}

params_clf = {
    'objective': 'binary',
    'metric': {'auc'}
}

params_reg = {
    'objective': 'regression_l1',
    'metric': {'l1'}
}


def get_params(details, run_type):
    # this function is not necessary to be put this way in the final submission here,
    # I had experiments to use LGB for classification, so I kept it this way for minimal code change
    params = params_base.copy()
    if run_type == 'clf':
        params.update(params_clf)
    elif run_type == 'reg':
        params.update(params_reg)
    else:
        raise Exception('unexpected run_type')
    params.update(details)
    return params


def pred_lgb_blend(test_x, gbms):
    raw_pred = []
    for gbm in gbms:
        raw_pred.append(gbm.predict(test_x))
    return np.array(raw_pred).mean(axis=0)


def pred_step2(x_raw, error1, mon_set, params):
    idx = x_raw['sale_month'].apply(lambda x: x in mon_set)
    y_step2 = error1[idx.values]
    x_raw_step2 = x_raw.loc[x_raw.index[idx], :]
    x_step2 = lgb_data_prep(x_raw_step2, keep_only_feature=step2_keep_only_feature)
    gbms_step2 = []
    for param in params:
        gbms_step2.append(train_lgb(x_step2, y_step2, get_params(param, 'reg')))

    # predict for 2016
    prop_2016_step2 = lgb_data_prep(prop_2016, keep_only_feature=step2_keep_only_feature)
    pred_2016_step2 = pred_lgb_blend(prop_2016_step2, gbms_step2)

    # predict for 2017
    prop_2017_step2 = lgb_data_prep(prop_2017, keep_only_feature=step2_keep_only_feature)
    pred_2017_step2 = pred_lgb_blend(prop_2017_step2, gbms_step2)
    return pred_2016_step2, pred_2017_step2


def lgb_pred_func():
    """2 step predict: raw + month"""
    # first raw prediction
    x_raw, y = rm_outlier(train_x, train_y)

    # raw prediction
    x_step1 = lgb_data_prep(x_raw, class3_new_features, class3_rm_features)
    gbms_step1 = []
    for param in raw_lgb_2y:
        gbms_step1.append(train_lgb(x_step1, y, get_params(param, 'reg')))
    pred1_train = pred_lgb_blend(x_step1, gbms_step1)
    error1 = y - pred1_train

    # step1 pred:
    prop_2016_step1 = lgb_data_prep(prop_2016, class3_new_features, class3_rm_features)
    pred_2016_step1 = pred_lgb_blend(prop_2016_step1, gbms_step1)

    prop_2017_step1 = lgb_data_prep(prop_2017, class3_new_features, class3_rm_features)
    pred_2017_step1 = pred_lgb_blend(prop_2017_step1, gbms_step1)

    # step2 pred
    # combined months pred
    pred_2016_step2, pred_2017_step2 = pred_step2(x_raw, error1, {'10', '11', '12'}, lgb_month)

    pred_2016 = pred_2016_step1 + pred_2016_step2
    pred_2017 = pred_2017_step1 + pred_2017_step2

    return pred_2016, pred_2017


# ------------------------------------------------------------ CatBoost prediction -----------------------------------------------------------------
# this essentially uses the script from the published kernel: https://www.kaggle.com/abdelwahedassklou/only-cat-boost-lb-0-0641-939
# but with several modifications:
# 1, apparently 2017 tax properties should be used for 2017 prediction, just not use it for 2016 prediction
# 2, I made use of categorical-transformation of census_block data as I did in lightGBM.
# 3, only month and quarter is used for seasonality adjustment,
# day should not be used as we don't have that in testing, and year-month and year-quarter is not used, as it is not usable for 2017 private LB prediction


from catboost import CatBoostRegressor  # version 0.2.5
from tqdm import tqdm


def catboost_pred_func():
    print('Loading Properties ...')
    properties2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    properties2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)

    print('Loading Train ...')
    train2016 = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
    train2017 = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

    def add_date_features(df):
        df["transaction_year"] = df["transactiondate"].dt.year
        df["transaction_month"] = df["transactiondate"].dt.month
        df["transaction_month"] = df["transaction_month"].apply(lambda x: x if x <= 10 else 10)
        df["transaction_quarter"] = df["transactiondate"].dt.quarter
        df.drop(["transactiondate"], inplace=True, axis=1)
        return df

    def proc_census_block(prop_data):
        prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
        prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('temp', axis=1, inplace=True)
        prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

        # census_block
        prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
        prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('censustractandblock', axis=1, inplace=True)
        return prop_data

    train2016 = add_date_features(train2016)
    train2017 = add_date_features(train2017)

    properties2016 = proc_census_block(properties2016)
    properties2017 = proc_census_block(properties2017)

    print('Loading Sample ...')
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)

    print('Merge Train with Properties ...')
    train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
    train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')

    print('Concat Train 2016 & 2017 ...')
    train_df = pd.concat([train2016, train2017], axis=0)
    test_df_2016 = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')
    test_df_2017 = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')

    del properties2016, properties2017, train2016, train2017
    gc.collect()

    print('Remove missing data fields ...')

    missing_perc_thresh = 0.98
    exclude_missing = []
    num_rows = train_df.shape[0]
    for c in train_df.columns:
        num_missing = train_df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing.append(c)
    print("We exclude: %s" % len(exclude_missing))

    del num_rows, missing_perc_thresh
    gc.collect()

    print("Remove features with one unique value !!")
    exclude_unique = []
    for c in train_df.columns:
        num_uniques = len(train_df[c].unique())
        if train_df[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            exclude_unique.append(c)
    print("We exclude: %s" % len(exclude_unique))

    print("Define training features !!")
    exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
    train_features = []
    for c in train_df.columns:
        if c not in exclude_missing \
                and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % len(train_features))

    print("Define categorial features !!")
    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        if c in {'census', 'block', 'raw_census', 'raw_block'}:
            cat_feature_inds.append(i)
        else:
            num_uniques = len(train_df[c].unique())
            if num_uniques < cat_unique_thresh \
                    and not 'sqft' in c \
                    and not 'cnt' in c \
                    and not 'nbr' in c \
                    and not 'number' in c:
                cat_feature_inds.append(i)

    print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
    print("all train features are: %s" % ','.join(train_features))

    print("Replacing NaN values by -999 !!")
    train_df.fillna(-999, inplace=True)
    test_df_2016.fillna(-999, inplace=True)
    test_df_2017.fillna(-999, inplace=True)

    print("Training time !!")
    X_train = train_df[train_features]
    y_train = train_df.logerror
    print(X_train.shape, y_train.shape)

    test_df_2016_10 = test_df_2016.copy()
    test_df_2016_10['transactiondate'] = pd.Timestamp('2016-10-01')
    test_df_2016_10 = add_date_features(test_df_2016_10)

    test_df_2017_10 = test_df_2017.copy()
    test_df_2017_10['transactiondate'] = pd.Timestamp('2017-10-01')
    test_df_2017_10 = add_date_features(test_df_2017_10)

    X_test_2016_10 = test_df_2016_10[train_features]
    X_test_2017_10 = test_df_2017_10[train_features]

    num_ensembles = 3
    y_pred_2016_10 = 0.0
    y_pred_2017_10 = 0.0

    models = []
    for i in tqdm(range(num_ensembles)):
        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=i)
        model.fit(
            X_train, y_train,
            cat_features=cat_feature_inds)
        models.append(model)
        y_pred_2016_10 += model.predict(X_test_2016_10)
        y_pred_2017_10 += model.predict(X_test_2017_10)

    y_pred_2016_10 /= num_ensembles
    y_pred_2017_10 /= num_ensembles

    return y_pred_2016_10, y_pred_2017_10


def blend_submission(fs, label):
    preds = [pd.read_csv('data/%s.csv.gz' % f, header=0, compression='gzip') for f in fs]
    columns = ['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']
    for p in preds:
        p = p.loc[:, columns]
    pred = None
    for p in preds:
        if pred is None:
            pred = p
        else:
            pred += p
    pred /= len(fs)
    pred['ParcelId'] = pred['ParcelId'].astype(int)
    pred.to_csv('data/%s.csv.gz' % label, index=False, float_format='%.7f', compression='gzip')


if __name__ == '__main__':
    prop_2016, prop_2017 = load_prop_data()
    train_x, train_y = load_train_data(prop_2016, prop_2017)

    pred_2016_lgb, pred_2017_lgb = lgb_pred_func()
    make_submission(pred_2016_lgb, pred_2017_lgb, 'final_lgb')

    del prop_2016, prop_2017, train_x, train_y
    gc.collect()

    pred_2016_catboost, pred_2017_catboost = catboost_pred_func()
    make_submission(pred_2016_catboost, pred_2017_catboost, 'final_catboost')

    # blend submission from intermediate csv files will produce exactly the same output as my chosen submission, as there is an application of  7-digits precision
    # directly do numerical average as above will introduce some difference at scale of 1e-7.
    blend_submission(['submission_final_lgb', 'submission_final_catboost'], 'submission_final')


