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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVR
import seaborn as sns
import params_cache as p
import data_cache

import pickle as pkl

prop_2016 = data_cache.prop_2016
prop_2017 = data_cache.prop_2017
train_x = data_cache.train_x
train_y = data_cache.train_y


def load_train_data_year(year):
    idx = (train_x['data_year'] == year).values
    x_year = train_x.loc[train_x.index[idx], :].copy()
    y_year = train_y[idx]
    return x_year, y_year


train_2017_x, train_2017_y = load_train_data_year(2017)
train_2016_x, train_2016_y = load_train_data_year(2016)


def load_train_data_2016(new_features=tuple()):
    for f in new_features:
        # group-by performed separately on 2016 and 2017 data
        feature_factory(f, prop_2016)

    train_2016 = pd.read_csv('data/train_2016_v2.csv')
    train = pd.merge(train_2016, prop_2016, how='left', on='parcelid')

    train['sale_month'] = train['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month
    train_y = train['logerror']
    train_x = train.drop('logerror', axis=1)
    return train_x, train_y


# train_2016_x, train_2016_y = load_train_data_2016()


from bayes_opt import BayesianOptimization
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

warnings.filterwarnings('ignore', category=DeprecationWarning)


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')
nmap_orig_to_new =  dict(zip(feature_info['orig_name'].values, feature_info.index.values))
nmap_new_to_orig =  dict(zip(feature_info.index.values, feature_info['orig_name'].values))

feature_imp_naive_lgb = pd.read_csv('records/feature_importance_raw_all.csv')


# KEY OBSERVATION: mean of (train_y - IS_pred_y) of lgb_raw with 2y blend model is ~0.007, meaning on average we are under estimating the error.
# naively adjust all predict up by 0.0035 boost LB score from 0.064406 to 0.064197 (ranking ~100).
# Yet, using MAE as measurement, adjust by mean does not improve score. MAE will only be improved by shifting by median.
# Since the training criterion is MAE, unsurprisingly, training pred-miss median is very close to zero, and validation sub sets also has pred-miss median close to zero. (train and validation are from shuffling here!)
# So the only explanation is that, LB testing set has a significant non-zero pred-miss median, meaning the distribution of the training set and testing are quite different.
# It is disturbing, if pred-miss median could be so volatile, then final private LB score is more of luck.

# if to train with non-10,11,12 month data, and use the ~8k 10,11,12 month data as test. We can observe a median of ~0.005 in pred-miss, meaning these 3 month has a different pattern from the others.
# it worth modeling the sale month in prediction, but sample size of those three month is too little.


def float_to_str(num):
    """float to 0_*** format"""
    s = '%.3f' % num
    return '_'.join(s.split('.'))


Y_Q_MAP = {'0_500': 0.006,
           '0_250': -0.0253,
           '0_750': 0.0392,
           '0_010': -0.3425,
           '0_990': 0.463882,
           '0_001': -1.158,
           '0_999': 1.6207}


y_outlier_down = -0.486476728279165
y_outlier_up = 0.7566


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


params_reg_naive = {
    'num_leaves': 40,
    'min_data_in_leaf': 300,
    'learning_rate': 0.01,
    'lambda_l2': 0.02,
    'num_boosting_rounds': 1500
}


params_clf_abs_error ={
    'num_leaves': 30,
    'min_data_in_leaf': 300,
    'learning_rate': 0.0035,
    'lambda_l2': 0.0045,
    'num_boosting_rounds': 2200
}

params_clf_sign_error = {
    'num_leaves': 40,
    'min_data_in_leaf': 280,
    'learning_rate': 0.0055,
    'lambda_l2': 0.01,
    'num_boosting_rounds': 1500
}

params_clf_mid_error = {
    'num_leaves': 30,
    'min_data_in_leaf': 220,
    'learning_rate': 0.004,
    'lambda_l2': 0.013,
    'num_boosting_rounds': 2500
}


params_reg_abs_error_big = {
    'num_leaves': 32,
    'min_data_in_leaf': 105,
    'learning_rate': 0.0075,
    'lambda_l2': 0.0121,
    'num_boosting_rounds': 1800
}


params_reg_abs_error_small = {
    'num_leaves': 28,
    'min_data_in_leaf': 100,
    'learning_rate': 0.0012,
    'lambda_l2': 0.05,
    'num_boosting_rounds': 1800
}


params_reg_sign_error_pos = {
    'num_leaves': 25,
    'min_data_in_leaf': 113,
    'learning_rate': 0.0026,
    'lambda_l2': 0.0014,
    'num_boosting_rounds': 1500
}


params_reg_sign_error_neg = {
    'num_leaves': 36,
    'min_data_in_leaf': 134,
    'learning_rate': 0.002,
    'lambda_l2': 0.041,
    'num_boosting_rounds': 1500
}


params_reg_mid_error_pos = {
    'num_leaves': 22,
    'min_data_in_leaf': 120,
    'learning_rate': 0.0016,
    'lambda_l2': 0.0026,
    'num_boosting_rounds': 1700
}


params_reg_mid_error_neg = {
    'num_leaves': 31,
    'min_data_in_leaf': 211,
    'learning_rate': 0.002,
    'lambda_l2': 0.0001,
    'num_boosting_rounds': 2700
}


# params_naive = {
#     'boosting_type': 'gbdt',
#     'objective': 'regression_l1',
#     'metric': {'l1'},
#     'num_leaves': 48,
#     'min_data_in_leaf': 200,
#     'learning_rate': 0.0045,
#     'lambda_l2': 0.004,
#     'feature_fraction': 0.8,
#     'bagging_fraction': 0.7,
#     'bagging_freq': 5,
#     'verbosity': 0,
#     'num_boosting_rounds': 2200
# }


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


def rm_outlier(x, y):
    idx = np.logical_and(y < 0.7566, y > -0.4865)
    x = x.loc[x.index[idx], :]
    y = y[idx]
    return x, y


def capfloor_outlier(x, y):
    y[y > 0.7566] = 0.7566
    y[y < -0.4865] = -0.4865
    return x, y


def get_params(details, run_type):
    params = params_base.copy()
    if run_type == 'clf':
        params.update(params_clf)
    elif run_type == 'reg':
        params.update(params_reg)
    else:
        raise Exception('unexpected run_type')
    params.update(details)
    return params


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
        # estimate = np.mean(train_data)
        estimate = np.median(train_data)
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


def pred_final(model, pred_2017=False):
    train_x, train_y = load_train_data(prop_2016, prop_2017)
    train_x_lgb = lgb_data_prep(train_x)
    prop_2016_lgb = lgb_data_prep(prop_2016)
    pred_2016 = model(train_x_lgb, train_y, prop_2016_lgb)

    pred_2017 = None
    if pred_2017:
        prop_2017_lgb = lgb_data_prep(prop_2017)
        pred_2017 = model(train_x_lgb, train_y, prop_2017_lgb)

    return pred_2016, pred_2017


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

    gbm = train_lgb(train_x, train_y, params)
    return gbm.predict(test_x)


def lgb_raw_2y_single_cv(train_x, train_y, test_x):

    train_x = pd.concat([train_x, train_2017_x], axis=0)
    train_y = pd.concat([train_y, train_2017_y])
    train_x_lgb = lgb_data_prep(train_x)
    test_x_lgb = lgb_data_prep(test_x)

    params = get_params(p.raw_lgb_2y_1, 'reg')
    gbm = train_lgb(train_x_lgb, train_y, params)
    out = gbm.predict(test_x_lgb)
    return out


def lgb_raw_2y_blend_cv(train_x, train_y, test_x):
    train_x = pd.concat([train_x, train_2017_x], axis=0)
    train_y = pd.concat([train_y, train_2017_y])
    train_x_lgb = lgb_data_prep(train_x)
    test_x_lgb = lgb_data_prep(test_x)

    preds = []
    for params_i in p.raw_lgb_2y:
        params = get_params(params_i, 'reg')
        gbm = train_lgb(train_x_lgb, train_y, params)
        pred = gbm.predict(test_x_lgb)
        preds.append(pred)

    return np.mean(np.array(preds), axis=0)


def lgb_raw_2y_blend_cv_class3_select(train_x, train_y, test_x):
    train_x = pd.concat([train_x, train_2017_x], axis=0)
    train_y = pd.concat([train_y, train_2017_y])
    train_x_lgb = lgb_data_prep(train_x, p.class3_new_features, p.class3_rm_features)
    test_x_lgb = lgb_data_prep(test_x, p.class3_new_features, p.class3_rm_features)

    preds = []
    for params_i in p.raw_lgb_2y:
        params = get_params(params_i, 'reg')
        gbm = train_lgb(train_x_lgb, train_y, params)
        pred = gbm.predict(test_x_lgb)
        preds.append(pred)

    return np.mean(np.array(preds), axis=0)


def lgb_raw_2y_blend_adj_mean_cv(train_x, train_y, test_x):
    train_x = pd.concat([train_x, train_2017_x], axis=0)
    train_y = pd.concat([train_y, train_2017_y])
    train_x_lgb = lgb_data_prep(train_x)
    test_x_lgb = lgb_data_prep(test_x)

    preds = []
    for params_i in p.raw_lgb_2y:
        params = get_params(params_i, 'reg')
        gbm = train_lgb(train_x_lgb, train_y, params)
        pred = gbm.predict(test_x_lgb)
        # pred_is = gbm.predict(train_x_lgb)
        # error = train_y - pred_is
        #
        # # use quantile range to get mean adj
        # idx = np.logical_or(error > error.quantile(0.75), error < error.quantile(0.25))
        # adj = error[~idx].mean()
        # print('IS mae: %.7f' % np.mean(np.abs(error)))
        #
        # print('n neg error: %.1f' % np.sum(error < 0))
        # print('n pos error: %.1f' % np.sum(error > 0))
        # print('error mean: %.7f' % error.mean())
        # print('error median: %.7f' % error.median())
        # print('error mid range mean: %.7f' % adj)
        # print('error up 0.25 range mean: %.7f' % error[error > error.quantile(0.75)].mean())
        # print('error down 0.25 range mean: %.7f' % error[error < error.quantile(0.25)].mean())
        #
        # adj = error.mean() * 0.3  # adj by half of mean, not too drastic
        # pred_final = pred + adj
        preds.append(pred)

    return np.mean(np.array(preds), axis=0)


def lgb_2layer_2y_cv(train_x, train_y, test_x):
    train_x = pd.concat([train_x, train_2017_x], axis=0)
    train_y = pd.concat([train_y, train_2017_y])
    train_x.index = list(range(train_x.shape[0]))
    train_y.index = train_x.index.copy()
    train_x_lgb = lgb_data_prep(train_x)
    test_x_lgb = lgb_data_prep(test_x)

    train_y_clf = np.zeros(train_y.shape[0])
    mark_1_idx = train_y > 0
    train_y_clf[mark_1_idx] = 1
    gbm_clf = train_lgb(train_x_lgb, train_y_clf, get_params(p.sign_clf_2y_1, 'clf'))
    prob_big = gbm_clf.predict(test_x_lgb)

    train_x_big_error = train_x_lgb.loc[train_x_lgb.index[mark_1_idx], :].copy()
    train_y_big_error = train_y[mark_1_idx].copy()
    gbm_big_error = train_lgb(train_x_big_error, train_y_big_error, get_params(p.sign_pos_2y_1, 'reg'))
    big_error_pred = gbm_big_error.predict(test_x_lgb)

    train_x_small_error = train_x_lgb.loc[train_x_lgb.index[~mark_1_idx], :].copy()
    train_y_small_error = train_y[~mark_1_idx].copy()
    gbm_small_error = train_lgb(train_x_small_error, train_y_small_error, get_params(p.sign_neg_2y_1, 'reg'))
    small_error_pred = gbm_small_error.predict(test_x_lgb)

    return big_error_pred * prob_big + (1 - prob_big) * small_error_pred


def lgb_raw_2y_blend_pred(train_x, train_y, test_x):

    train_x_lgb = lgb_data_prep(train_x)
    test_x_lgb = lgb_data_prep(test_x)

    preds = []
    for params_i in p.raw_lgb_2y:
        params = get_params(params_i, 'reg')
        gbm = train_lgb(train_x_lgb, train_y, params)
        pred = gbm.predict(test_x_lgb)
        preds.append(pred)

    return np.mean(np.array(preds), axis=0)


def lgb_raw_2y_blend_pred_class3_select(train_x, train_y, test_x):

    train_x_lgb = lgb_data_prep(train_x, p.class3_new_features, p.class3_rm_features)
    test_x_lgb = lgb_data_prep(test_x, p.class3_new_features, p.class3_rm_features)

    preds = []
    for params_i in p.raw_lgb_2y:
        params = get_params(params_i, 'reg')
        gbm = train_lgb(train_x_lgb, train_y, params)
        pred = gbm.predict(test_x_lgb)
        preds.append(pred)

    return np.mean(np.array(preds), axis=0)


def lgb_raw_2y_blend_adj_mean_pred(train_x, train_y, test_x):
    train_x = pd.concat([train_x, train_2017_x], axis=0)
    train_y = pd.concat([train_y, train_2017_y])
    train_x_lgb = lgb_data_prep(train_x)
    test_x_lgb = lgb_data_prep(test_x)

    preds = []
    for params_i in p.raw_lgb_2y:
        params = get_params(params_i, 'reg')
        gbm = train_lgb(train_x_lgb, train_y, params)
        pred = gbm.predict(test_x_lgb)
        pred_is = gbm.predict(train_x_lgb)
        error = train_y - pred_is
        adj = error.mean() * 0.3  # adj by half of mean, not too drastic
        pred_final = pred + adj
        preds.append(pred_final)

    return np.mean(np.array(preds), axis=0)


def outlier_handling(train_data, test_data):
    # set extreme x to na
    train_data, _ = outlier_x_clean(train_data, train_data['logerror'], test_data, type_id='na', thresh=0.001)
    # remove extreme y from train
    train_data, _ = outlier_y_rm(train_data, train_data['logerror'], None, 0.001)
    return train_data, test_data


def outlier_handling_cv(train_x, train_y, test_data, thresh):
    # set extreme x to na
    # train_x, train_y = outlier_x_clean(train_x, train_y, test_data, type_id='na', thresh=thresh)
    # remove extreme y from train
    train_x, train_y = outlier_y_rm(train_x, train_y, None, thresh)
    return train_x, train_y


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


def outlier_x_clean(train_x_inp, train_y_inp, test_x, type_id='na', thresh=0.001):
    """type_id:
       na: fill outliers with na for both train and test
       mean: fill outlier with mean for both train and test
       rm: remove outlier from train
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
        if type_id == 'na':
            test_x.loc[idx_test, col] = np.nan

    def proc_outlier_cat(col):
        idx_train = train_x[col].apply(TYPE_VAR_CLEAN_MAP[col])
        idx_test = test_x[col].apply(TYPE_VAR_CLEAN_MAP[col])
        rm_idx_train[col] = idx_train
        train_x.loc[idx_train, col] = np.nan
        if type_id == 'na':
            test_x.loc[idx_test, col] = np.nan

    # year built use left side only
    proc_outlier_num('year_built', thresh, -1)

    # all others use two-sided
    for num_var in ('area_lot', 'dollar_tax', 'area_living_type_12', 'dollar_taxvalue_structure',
                    'area_living_finished_calc', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
                    'area_garage', 'area_living_type_15', 'area_pool'):
        if num_var in train_x_inp.columns:
            proc_outlier_num(num_var, thresh, 1-thresh)

    for cat_var in TYPE_VAR_CLEAN_MAP:
        if cat_var in ('type_heating_system', 'type_landuse'):
            # type variables have already been coded to int in load_data_naive_lgb
            continue
        if cat_var in train_x_inp.columns:
            proc_outlier_cat(cat_var)

    if type_id == 'rm':
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


def property_trans(prop_data):
    """only does transformation, no cleaning"""

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

    # for name in feature_info.index.values:
    #     if name[:4] == 'num_':
    #         cat_num_to_str(prop_data, nmap_new_to_orig[name])
    #         feature_info.loc[name, 'type'] = 'cat'

    # name columns
    prop_data.rename(columns=nmap_orig_to_new, inplace=True)


TYPE_VAR_CLEAN_MAP_OUTLIER = {
    'type_air_conditioning': lambda x: x in ['11', '9', '3'],
    'type_architectural_style': lambda x: x in ['10'],
    'area_base_finished': lambda x: x > 1600,
    'num_bathroom_assessor': lambda x: x >= 9,
    'num_bathroom_zillow': lambda x: x >= 9,
    'num_bedroom': lambda x: x >= 11,
    'rank_building_quality': lambda x: x in (2,),
    'area_firstfloor_zillow': lambda x: x > 5000,
    'area_living_finished_calc': lambda x: x > 15000,
    'area_living_type_12': lambda x: x > 15000,
    'area_living_type_15': lambda x: x > 10000,
    'area_firstfloor_assessor': lambda x: x > 6500,
    'area_living_type_6': lambda x: x > 5500,
    'num_fireplace': lambda x: x ==  5,
    'num_fullbath': lambda x: x >= 11,
    'num_garage': lambda x: x >= 6,
    'area_garage': lambda x: x > 3500,
    'type_heating_system': lambda x: x in ('11', '14', '12', '10'),

    'type_landuse': lambda x: x in ('260', '263', '264', '267', '275', '31', '47'),
    'num_room': lambda x: x > 11 or (x < 3 and x != 0),
    'num_34_bathroom': lambda x: x >= 2,
    'num_unit': lambda x: x >= 5,
    'num_story': lambda x: x >= 4,
}


def property_cleaning_v2(prop_data):
    """clean cols according to train_data distribution"""

    def clear_cat_col_group(col, groups):
        """set given groups of a categorical col to na"""
        in_group_flag = prop_data[col].apply(lambda x: x in groups)
        prop_data.loc[prop_data[col].index[in_group_flag], col] = np.nan

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
TYPE_VAR_CLEAN_MAP = {
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
        new_col_data[new_col_data.index[new_col_data.apply(TYPE_VAR_CLEAN_MAP[col])]] = to_val
        data[new_col_name] = new_col_data
        cat_num_to_str(data, new_col_name)

    new_col_name = 'groupby__' + col

    if col in ('type_air_conditioning', 'flag_pool', 'flag_tax_delinquency',
               'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
               'raw_block', 'raw_census', 'block', 'census', 'code_fips'):
        # already grouped in basic cleaning
        data[new_col_name] = data[col].copy()
    else:
        data[new_col_name] = data[col].copy()
        cat_num_to_str(data, new_col_name)

    # if col == 'num_bathroom_zillow':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_bedroom':
    #     clean_type_var(np.nan)
    #
    # if col == 'rank_building_quality':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_fireplace':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_fullbath':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_garage':
    #     clean_type_var(np.nan)
    #
    # if col == 'type_heating_system':
    #     clean_type_var(np.nan)
    #
    # if col == 'type_landuse':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_room':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_34_bathroom':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_unit':
    #     clean_type_var(np.nan)
    #
    # if col == 'num_story':
    #     clean_type_var(np.nan)

    return new_col_name


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

    col_diff_dollar_taxvalue_total = prop_2017['dollar_taxvalue_total'] / prop_2016['dollar_taxvalue_total']
    prop_2016['2y_diff_dollar_taxvalue_total'] = col_diff_dollar_taxvalue_total
    prop_2017['2y_diff_dollar_taxvalue_total'] = col_diff_dollar_taxvalue_total
    col_diff_dollar_taxvalue_land = prop_2017['dollar_taxvalue_land'] / prop_2016['dollar_taxvalue_land']
    prop_2016['2y_diff_dollar_taxvalue_land'] = col_diff_dollar_taxvalue_land
    prop_2017['2y_diff_dollar_taxvalue_land'] = col_diff_dollar_taxvalue_land
    col_diff_dollar_taxvalue_structure = prop_2017['dollar_taxvalue_structure'] / prop_2016['dollar_taxvalue_structure']
    prop_2016['2y_diff_dollar_taxvalue_structure'] = col_diff_dollar_taxvalue_structure
    prop_2017['2y_diff_dollar_taxvalue_structure'] = col_diff_dollar_taxvalue_structure

    for f in p.class3_new_features:
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
    basedate = pd.to_datetime('2015-11-15').toordinal()
    train['cos_season'] = ((pd.to_datetime(train['transactiondate']).apply(lambda x: x.toordinal() - basedate)) * (2 * np.pi / 365.25)).apply(np.cos)
    train['sin_season'] = ((pd.to_datetime(train['transactiondate']).apply(lambda x: x.toordinal() - basedate)) * (2 * np.pi / 365.25)).apply(np.sin)
    train_y = train['logerror']
    train_x = train.drop('logerror', axis=1)
    return train_x, train_y


def prop_compare(prop_2016, prop_2017):
    print('2016 shape: ' + str(prop_2016.shape))
    print('2017 shape: ' + str(prop_2017.shape))
    n_sample = prop_2016.shape[0]
    out = []
    for col in prop_2016:
        if col == 'parcelid':
            continue
        missing_2016 = np.sum(prop_2016[col].isnull()) / n_sample
        missing_2017 = np.sum(prop_2017[col].isnull()) / n_sample
        # disagreed rows:
        diff_nan = np.sum(np.logical_xor(prop_2016[col].isnull(), prop_2017[col].isnull())) / n_sample
        non_val_idx = np.logical_and(~prop_2016[col].isnull(), ~prop_2017[col].isnull())
        diff_val = np.sum(prop_2016.loc[non_val_idx, col] != prop_2017.loc[non_val_idx, col]) / n_sample
        out.append([col, missing_2016, missing_2017, diff_nan, diff_val])
        print('%s: 2016 missing: %.6f; 2017 missing: %.6f; nan_diff: %.6f; val_diff: %.6f' % (col, missing_2016, missing_2017, diff_nan, diff_val))
    df = pd.DataFrame(out, columns=['col', 'missing_2016', 'missing_2017', 'diff_nan', 'diff_val'])
    df.to_csv('data/compare_2016_2017.csv')


def load_data_raw():
    # init load data
    prop_data = pd.read_csv('data/properties_2016.csv', header=0)
    error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
    error_data['sale_month'] = error_data['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month
    property_cleaning(prop_data)

    for col in prop_data.columns:
        if prop_data[col].dtype == np.float64:
            prop_data.loc[:, col] = prop_data[col].astype(np.float32)
    prop_data[['latitude', 'longitude']] /= 1e6
    prop_data[['latitude', 'longitude']] /= 1e6

    features_f = open('features_raw_lgb.txt', 'r')
    f_list = [f for f in features_f.read().split('\n') if f]
    features_f.close()

    for f in f_list:
        feature_factory(f, prop_data)

    train_data = error_data.merge(prop_data, how='left', on='parcelid')
    # train_data.to_csv('data/train_data_merge.csv', index=False)

    submission = pd.read_csv('data/sample_submission.csv', header=0)
    submission['parcelid'] = submission['ParcelId']
    submission.drop('ParcelId', axis=1, inplace=True)
    test_data = submission.merge(prop_data, how='left', on='parcelid')

    clean_class3_var(train_data, test_data)

    # raw feature filtering
    keep_feature = list(feature_info.index.values) + f_list
    for col in ['census_block', 'raw_census_block', 'year_assess']:
        keep_feature.remove(col)
    test_x = test_data[keep_feature]
    train_x = train_data[keep_feature]
    train_y = train_data['logerror']

    # train_x, train_y = outlier_x_clean(train_x, train_y, test_x, type_id='na', thresh=0.001)
    return train_x, train_y, test_x


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

    col_new = col + '_lgb'
    # create and use map from test data only
    uni_vals = np.sort(test_data[col].unique()).tolist()
    m = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
    train_data[col_new] = train_data[col].apply(lambda x: m[x])
    train_data[col_new] = train_data[col_new].astype('category')

    test_data[col_new] = test_data[col].apply(lambda x: m[x])
    test_data[col_new] = test_data[col_new].astype('category')


def convert_cat_col_single(data, col):
    # convert to lgb usable categorical
    # set nan to string so that can be sorted, make sure training set and testing set get the same coding
    data.loc[data.index[data[col].isnull()], col] = 'nan'
    uni_vals = np.sort(data[col].unique()).tolist()
    map = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
    col_new = col + '_lgb'
    data[col_new] = data[col].apply(lambda x: map[x])
    data[col_new] = data[col_new].astype('category')


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
        if col in keep_feature:
            keep_feature.remove(col)

    if len(keep_only_feature) > 0:
        keep_feature = list(keep_only_feature)

    return data[keep_feature]


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


def train_lgb_with_val(train_x, train_y, params_inp):
    # train - validaiton split
    train_x_use, val_x, train_y_use, val_y = train_test_split(train_x, train_y, test_size=0.3, random_state=42)

    # create lgb dataset
    lgb_train = lgb.Dataset(train_x_use, train_y_use)
    lgb_val = lgb.Dataset(val_x, val_y, reference=lgb_train)

    params = params_inp.copy()
    if 'num_boosting_rounds' in params:
        params.pop('num_boosting_rounds')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=3000,
                    valid_sets=lgb_val,
                    early_stopping_rounds=100)

    return gbm


def train_lgb(train_x, train_y, params_inp):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = params_inp.copy()
    num_boost_round = params.pop('num_boosting_rounds')
    gbm = lgb.train(params, lgb_train, num_boost_round=num_boost_round)
    return gbm


def cv_lgb_final(train_x, train_y, params_inp):
    lgb_train = lgb.Dataset(train_x, train_y)
    params = params_inp.copy()
    num_boost_round = params.pop('num_boosting_rounds')
    eval_hist = lgb.cv(params, lgb_train, stratified=False, num_boost_round=num_boost_round, early_stopping_rounds=30)
    return eval_hist['l1-mean'][-1], eval_hist['l1-stdv'][-1], len(eval_hist['l1-mean'])


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
    return features_sorted, rank_sorted


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


def search_lgb_bo(train_x, train_y, params, label='', n_iter=80,
                  min_data_in_leaf_range=(200, 800),
                  num_leaf_range=(30, 80),
                  do_clf=False):
    """for mae target, need to flip sign to maximize.
       for auc, no need to flip sign"""
    lgb_train = lgb.Dataset(train_x, train_y)

    def lgb_evaluate(num_leaves,
                     min_data_in_leaf,
                     learning_rate_log,
                     # lambda_l2_log
                     ):
        learning_rate = 0.1 ** learning_rate_log
        # lambda_l2 = 0.1 ** lambda_l2_log
        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        params.update({'num_leaves': num_leaves,
                       'min_data_in_leaf': min_data_in_leaf,
                       'learning_rate': learning_rate,})

        eval_hist = lgb.cv(params, lgb_train, stratified=False, num_boost_round=3000, early_stopping_rounds=30)
        if do_clf:
            return eval_hist['auc-mean'][-1]
        else:
            return -eval_hist['l1-mean'][-1]

    search_range = {'num_leaves': (num_leaf_range[0], num_leaf_range[1]),
                    'min_data_in_leaf': (min_data_in_leaf_range[0], min_data_in_leaf_range[1]),
                    'learning_rate_log': (1, 3),
                    # 'lambda_l2_log': (1, 4)
                    }
    lgb_bo = BayesianOptimization(lgb_evaluate, search_range)
    lgb_bo.maximize(n_iter=n_iter, init_points=5)

    res = lgb_bo.res['all']
    res_df = pd.DataFrame()
    res_df['score'] = -np.array(res['values'])
    for v in search_range:
        if v in ('learning_rate_log',
                 #'lambda_l2_log'
                 ):
            v_label = v[:-4]
            apply_func = lambda x: 0.1 ** x
        else:
            v_label = v
            apply_func = lambda x: x
        res_df[v_label] = np.array([apply_func(d[v]) for d in res['params']])
    res_df.to_csv('temp_cv_res_bo_%s.csv' % label, index=False)
    print('BO search finished')


def search_lgb_random(train_x, train_y, params, label='', n_iter=80,
                      min_data_in_leaf_range=(200, 800),
                      num_leaf_range=(30, 80),
                      with_rm_outlier=False):
    lgb_train = lgb.Dataset(train_x, train_y)
    if 'num_boosting_rounds' in params:
        params.pop('num_boosting_rounds')

    metric = list(params['metric'])[0]
    if with_rm_outlier:
        train_x_outlier_rm, train_y_outlier_rm = rm_outlier(train_x, train_y)
        lgb_train_outlier = lgb.Dataset(train_x_outlier_rm, train_y_outlier_rm)
        columns = ['%s-mean' % metric, '%s-stdv' % metric, '%s-mean_outlier_rm' % metric, '%s-stdv' % metric, 'n_rounds', 'num_leaves', 'min_data_in_leaf', 'learning_rate',
                   # 'lambda_l2'
                   ]
    else:
        columns = ['%s-mean' % metric, '%s-stdv' % metric, 'n_rounds', 'num_leaves', 'min_data_in_leaf', 'learning_rate',
        # 'lambda_l2'
        ]
    headers = ','.join(columns)

    def rand_min_data_in_leaf():
        return np.random.randint(min_data_in_leaf_range[0], min_data_in_leaf_range[1])

    def rand_learning_rate():
        return np.random.uniform(1, 3)

    def rand_num_leaf():
        return np.random.randint(num_leaf_range[0], num_leaf_range[1])

    def rand_lambda_l2():
        return np.random.uniform(1, 4)

    def write_to_file(line):
        f = open('temp_cv_res_random_%s.txt' % label, 'a')
        f.write(line + '\n')
        f.close()

    res = []
    write_to_file(headers)
    for i in range(1, n_iter + 1):
        rand_params = {'num_leaves': rand_num_leaf(),
                       'min_data_in_leaf': rand_min_data_in_leaf(),
                       'learning_rate': 0.1 ** rand_learning_rate(),
                       # 'lambda_l2': 0.1 ** rand_lambda_l2()
                       }
        params.update(rand_params)
        eval_hist = lgb.cv(params, lgb_train, stratified=False, num_boost_round=10000, early_stopping_rounds=100)
        if with_rm_outlier:
            eval_hist_outlier_rm = lgb.cv(params, lgb_train_outlier, stratified=False, num_boost_round=6000, early_stopping_rounds=100)
            res_list = [eval_hist['%s-mean' % metric][-1],
                        eval_hist['%s-stdv' % metric][-1],
                        eval_hist_outlier_rm['%s-mean' % metric][-1],
                        eval_hist_outlier_rm['%s-stdv' % metric][-1],
                        len(eval_hist_outlier_rm['%s-mean' % metric]),
                        rand_params['num_leaves'],
                        rand_params['min_data_in_leaf'],
                        rand_params['learning_rate'],
                        # rand_params['lambda_l2']
                        ]

        else:
            res_list = [eval_hist['%s-mean' % metric][-1],
                         eval_hist['%s-stdv' % metric][-1],
                         len(eval_hist['%s-mean' % metric]),
                         rand_params['num_leaves'],
                         rand_params['min_data_in_leaf'],
                         rand_params['learning_rate'],
                         # rand_params['lambda_l2']
                         ]
            write_to_file('%.7f,%.7f,%.0f,%.0f,%.0f,%.6f' % tuple(res_list))
        res.append(res_list)

        print('finished %d / %d' % (i, n_iter))
    res_df = pd.DataFrame(res, columns=columns)
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
                    eval_hist = lgb.cv(params, lgb_train, stratified=False, num_boost_round=3000, early_stopping_rounds=30)
                    res.append([eval_hist['l1-mean'][-1],
                                num_leaf,
                                min_data_in_leaf,
                                learning_rate,
                                lambda_l2])
                    iter += 1
                    print('finished %d / %d' % (iter, n_trial))
    res_df = pd.DataFrame(res, columns=['score', 'num_leaves', 'min_data_in_leaf', 'learning_rate', 'lambda_l2'])
    res_df.to_csv('records/temp_cv_res_grid.csv', index=False)


def submit_nosea(score, ver):
    submission = pd.read_csv('data/sample_submission.csv', header=0)
    df = pd.DataFrame()
    df['ParcelId'] = submission['ParcelId']
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


# NUM_VARS = ['area_lot', 'area_living_type_12', 'dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
#             'area_garage', 'area_pool']
#
# CAT_VARS = ['type_air_conditioning', 'flag_pool', 'flag_tax_delinquency', 'type_heating_system', 'type_landuse',
#             'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
#             'raw_block', 'raw_census', 'block', 'census',
#             'num_bathroom_zillow', 'num_bedroom', 'rank_building_quality', 'code_fips', 'num_fireplace', 'num_fullbath',
#             'num_garage', 'num_room', 'num_unit', 'num_story']


NUM_VARS = ['area_lot', 'area_living_type_12', 'dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
            'area_garage', 'area_pool']

CAT_VARS = ['str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
            'raw_block', 'raw_census', 'block', 'census',
            'num_bedroom', 'rank_building_quality', 'num_room', 'num_unit']

# NUM_VARS = ['area_living_type_12']
#
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
    elif op_type == 'mean':
        new_col = num_col + '__' + new_cat_var + '__mean'
        data[new_col] = avg_col
    else:
        raise Exception('unknown op type')

    # rm small group idx
    data.loc[rm_idx, new_col] = np.nan
    # clear newly created col
    test_data.drop(new_cat_var, axis=1, inplace=True)
    raw_data.drop(new_cat_var, axis=1, inplace=True)

    return new_col


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
    train_x_fe = lgb_data_prep(train_x_fe)

    num_vars = NUM_VARS + new_num_vars
    # num_vars = NUM_VARS

    for cat_var in CAT_VARS:
        old_features = list(train_x_fe.columns)
        new_features = groupby_feature_gen_batch(num_vars, cat_var, train_x_fe, train_x_fe_raw, test_x_fe)
        gbm = train_lgb(train_x_fe, train_y)
        features_sorted, _ = feature_importance(gbm, 'fe1_after_%s' % cat_var, print_to_scr=False)
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
    train_x_fe = lgb_data_prep(train_x_fe)

    num_vars = NUM_VARS + new_num_vars
    # num_vars = NUM_VARS

    for cat_var in CAT_VARS:
        old_features = list(train_x_fe.columns)
        new_features = groupby_feature_gen_batch(num_vars, cat_var, train_x_fe, train_x_fe_raw, test_x_fe)
        gbm = train_lgb(train_x_fe, train_y)
        features_sorted, _ = feature_importance(gbm, 'fe2_after_%s' % cat_var, print_to_scr=False)
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


def feature_engineering3(train_x_inp, train_y, params, label):

    f_name = 'fe_%s.txt' % label

    def write_to_file(out_str):
        f = open(f_name, 'a')
        f.write(out_str)
        f.close()

    def fe_cv(data, col_name):
        gbm = train_lgb(data, train_y)
        feature_sorted, avg_rank_sorted = feature_importance(gbm, print_to_scr=False)
        col_rank = avg_rank_sorted[list(feature_sorted).index(col_name)]
        cv_mean, cv_stdv, _ = cv_lgb_final(data, train_y, params)
        write_to_file('%s,%.7f,%.7f,%.1f\n' % (col_name, cv_mean, cv_stdv, col_rank))

    train_x = train_x_inp.copy()
    prep_for_lgb_single(train_x)
    train_x_lgb = lgb_data_prep(train_x)

    write_to_file('col,score_mean,score_stdv,avg_rank\n')

    # raw lgb
    cv_mean, cv_stdv, _ = cv_lgb_final(train_x_lgb, train_y, params)
    write_to_file('%s,%.7f,%.7f,%.1f\n' % ('None', cv_mean, cv_stdv, 0))

    new_num_features = []

    # try each of the engineered features
    areas = ('area_lot', 'area_garage', 'area_pool', 'area_living_type_12', 'area_living_type_15')
    num_vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total')
    # areas = ('area_lot',)
    # num_vars = ('dollar_tax',)

    train_x_lgb['dollar_taxvalue_structure_land_diff'] = train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land']
    train_x_inp['dollar_taxvalue_structure_land_diff'] = train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land']
    new_num_features.append('dollar_taxvalue_structure_land_diff')
    fe_cv(train_x_lgb, 'dollar_taxvalue_structure_land_diff')
    train_x_lgb.drop('dollar_taxvalue_structure_land_diff', axis=1, inplace=True)

    train_x_lgb['dollar_taxvalue_structure_land_absdiff'] = np.abs(train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land'])
    train_x_inp['dollar_taxvalue_structure_land_absdiff'] = np.abs(train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land'])
    new_num_features.append('dollar_taxvalue_structure_land_absdiff')
    fe_cv(train_x_lgb, 'dollar_taxvalue_structure_land_absdiff')
    train_x_lgb.drop('dollar_taxvalue_structure_land_absdiff', axis=1, inplace=True)

    train_x_lgb['dollar_taxvalue_structure_land_diff_norm'] = (train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land']) / train_x_inp['dollar_taxvalue_total']
    train_x_inp['dollar_taxvalue_structure_land_diff_norm'] = (train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land']) / train_x_inp['dollar_taxvalue_total']
    new_num_features.append('dollar_taxvalue_structure_land_diff_norm')
    fe_cv(train_x_lgb, 'dollar_taxvalue_structure_land_diff_norm')
    train_x_lgb.drop('dollar_taxvalue_structure_land_diff_norm', axis=1, inplace=True)

    train_x_lgb['dollar_taxvalue_structure_land_absdiff_norm'] = np.abs(train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land']) / train_x_inp['dollar_taxvalue_total']
    train_x_inp['dollar_taxvalue_structure_land_absdiff_norm'] = np.abs(train_x_inp['dollar_taxvalue_structure'] - train_x_inp['dollar_taxvalue_land']) / train_x_inp['dollar_taxvalue_total']
    new_num_features.append('dollar_taxvalue_structure_land_absdiff_norm')
    fe_cv(train_x_lgb, 'dollar_taxvalue_structure_land_absdiff_norm')
    train_x_lgb.drop('dollar_taxvalue_structure_land_absdiff_norm', axis=1, inplace=True)

    train_x_lgb['dollar_taxvalue_structure_total_ratio'] = train_x_inp['dollar_taxvalue_structure'] / train_x_inp['dollar_taxvalue_total']
    train_x_inp['dollar_taxvalue_structure_total_ratio'] = train_x_inp['dollar_taxvalue_structure'] / train_x_inp['dollar_taxvalue_total']
    new_num_features.append('dollar_taxvalue_structure_total_ratio')
    fe_cv(train_x_lgb, 'dollar_taxvalue_structure_total_ratio')
    train_x_lgb.drop('dollar_taxvalue_structure_total_ratio', axis=1, inplace=True)

    train_x_lgb['dollar_taxvalue_total_dollar_tax_ratio'] = train_x_inp['dollar_taxvalue_total'] / train_x_inp['dollar_tax']
    train_x_inp['dollar_taxvalue_total_dollar_tax_ratio'] = train_x_inp['dollar_taxvalue_total'] / train_x_inp['dollar_tax']
    new_num_features.append('dollar_taxvalue_total_dollar_tax_ratio')
    fe_cv(train_x_lgb, 'dollar_taxvalue_total_dollar_tax_ratio')
    train_x_lgb.drop('dollar_taxvalue_total_dollar_tax_ratio', axis=1, inplace=True)

    train_x_lgb['living_area_proportion'] = train_x_inp['area_living_type_12'] / train_x_inp['area_lot']
    train_x_inp['living_area_proportion'] = train_x_inp['area_living_type_12'] / train_x_inp['area_lot']
    new_num_features.append('living_area_proportion')
    fe_cv(train_x_lgb, 'living_area_proportion')
    train_x_lgb.drop('living_area_proportion', axis=1, inplace=True)

    # per_square variables
    for v in num_vars:
        for a in areas:
            col_name = v + '__per__' + a
            train_x_lgb[col_name] = train_x_inp[v] / train_x_inp[a]
            train_x_inp[col_name] = train_x_inp[v] / train_x_inp[a]
            new_num_features.append(col_name)
            train_x_lgb.loc[np.abs(train_x_inp[a]) < 1e-5, col_name] = np.nan
            fe_cv(train_x_lgb, col_name)
            train_x_lgb.drop(col_name, axis=1, inplace=True)

    # groupby features
    num_features = new_num_features + NUM_VARS
    # num_features = NUM_VARS
    for cat_var in CAT_VARS:
        for num_var in num_features:
            for op_type in ('neu', 'absneu', 'mean'):
                col_name = groupby_feature_gen_single(num_var, cat_var, train_x_lgb, train_x_inp, op_type)
                fe_cv(train_x_lgb, col_name)
                train_x_lgb.drop(col_name, axis=1, inplace=True)


def feature_engineering3_combined():
    """group by features should be engineered from prop data"""
    f_name = 'fe_2y_raw_lgb_rank.txt'
    params = get_params(p.raw_lgb_2y_2, 'reg')

    f_file = open(f_name, 'r')
    lines = f_file.read().split('\n')
    cols = [l.split(',')[0] for l in lines]

    def write_to_file(out_str):
        f = open(f_name, 'a')
        f.write(out_str)
        f.close()

    def cv(col, keep_col=False):
        if col in cols:
            return
        print('processing: %s' % col)
        keep_num_var = True if '__groupby__' in col else False
        feature_factory(col, prop_2016, keep_num_var)
        feature_factory(col, prop_2017, keep_num_var)
        train_x, train_y = load_train_data(prop_2016, prop_2017)
        train_x_lgb = lgb_data_prep(train_x, (col,))
        print('train_x shape: %s' % str(train_x_lgb.shape))

        gbm = train_lgb(train_x_lgb, train_y, params)
        feature_sorted, avg_rank_sorted = feature_importance(gbm, print_to_scr=False)
        col_rank = avg_rank_sorted[list(feature_sorted).index(col)]
        # cv_mean, cv_stdv, _ = cv_lgb_final(train_x_lgb, train_y, params)
        # write_to_file('%s,%.7f,%.7f,%.1f\n' % (col, cv_mean, cv_stdv, col_rank))
        write_to_file('%s,%.7f,%.7f,%.1f\n' % (col, 0.0, 0.0, col_rank))

        if not keep_col:
            prop_2016.drop(col, axis=1, inplace=True)
            prop_2017.drop(col, axis=1, inplace=True)

    write_to_file('col,score_mean,score_stdv,avg_rank\n')

    # raw lgb
    train_x, train_y = load_train_data(prop_2016, prop_2017)
    train_x_lgb = lgb_data_prep(train_x)
    cv_mean, cv_stdv, _ = cv_lgb_final(train_x_lgb, train_y, params)
    write_to_file('%s,%.7f,%.7f,%.1f\n' % ('None', cv_mean, cv_stdv, 0))

    new_num_features = []

    # try each of the engineered features
    areas = ('area_lot', 'area_garage', 'area_pool', 'area_living_type_12', 'area_living_type_15')
    num_vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total')
    # areas = ('area_lot',)
    # num_vars = ('dollar_tax',)

    for feature in ['dollar_taxvalue_structure_land_diff_norm',
                    'dollar_taxvalue_structure_land_absdiff_norm',
                    'dollar_taxvalue_structure_total_ratio',
                    'dollar_taxvalue_total_dollar_tax_ratio',
                    'living_area_proportion']:
        new_num_features.append(feature)
        cv(feature, True)

    # per_square variables
    for v in num_vars:
        for a in areas:
            col_name = v + '__per__' + a
            new_num_features.append(col_name)
            cv(col_name)

    # groupby features
    num_features = new_num_features + NUM_VARS
    # num_features = NUM_VARS
    for cat_var in CAT_VARS:
        for num_var in num_features:
            for op_type in ('mean', 'neu', 'absneu'):
                col_name = num_var + '__groupby__' + cat_var + '__' + op_type
                cv(col_name)
            if cat_var == CAT_VARS[0] and num_var in num_features[0]:
                # only do count for one of them
                col_name = num_var + '__groupby__' + cat_var + '__count'
                cv(col_name, True)


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


# # -------------------------------------------------------------- 2layer ------------------------------------------------------------
# def pred_2layer_abs_clf(train_x, train_y, test_x):
#
#     params_clf_local = params_base.copy()
#     params_reg_local = params_base.copy()
#     params_clf_local.update(params_clf)
#     params_reg_local.update(params_reg)
#
#     params_clf_local.update(params_clf_abs_error)
#     params_reg_big = params_reg_local.copy()
#     params_reg_big.update(params_reg_abs_error_big)
#     params_reg_small = params_reg_local.copy()
#     params_reg_small.update(params_reg_abs_error_small)
#
#     train_y_clf = np.zeros(train_y.shape[0])
#     mark_1_idx = np.logical_or(train_y > Y_Q_MAP[float_to_str(0.75)], train_y < Y_Q_MAP[float_to_str(0.25)])
#     train_y_clf[mark_1_idx] = 1
#     gbm_clf = train_lgb(train_x, train_y_clf, params_clf_local)
#     prob_big = gbm_clf.predict(test_x)
#
#     train_x_big_error = train_x.loc[train_x.index[mark_1_idx], :].copy()
#     train_y_big_error = train_y[mark_1_idx].copy()
#     gbm_big_error = train_lgb(train_x_big_error, train_y_big_error, params_reg_big)
#     big_error_pred = gbm_big_error.predict(test_x)
#
#     train_x_small_error = train_x.loc[train_x.index[~mark_1_idx], :].copy()
#     train_y_small_error = train_y[~mark_1_idx].copy()
#     gbm_small_error = train_lgb(train_x_small_error, train_y_small_error, params_reg_small)
#     small_error_pred = gbm_small_error.predict(test_x)
#
#     return big_error_pred * prob_big + (1 - prob_big) * small_error_pred
#
#
# def pred_2layer_sign_clf(train_x, train_y, test_x):
#
#     params_clf_local = params_base.copy()
#     params_reg_local = params_base.copy()
#     params_clf_local.update(params_clf)
#     params_reg_local.update(params_reg)
#
#     params_clf_local.update(params_clf_sign_error)
#     params_reg_big = params_reg_local.copy()
#     params_reg_big.update(params_reg_sign_error_pos)
#     params_reg_small = params_reg_local.copy()
#     params_reg_small.update(params_reg_sign_error_neg)
#
#     train_y_clf = np.zeros(train_y.shape[0])
#     mark_1_idx = train_y > 0
#     train_y_clf[mark_1_idx] = 1
#     gbm_clf = train_lgb(train_x, train_y_clf, params_clf_local)
#     prob_big = gbm_clf.predict(test_x)
#
#     train_x_big_error = train_x.loc[train_x.index[mark_1_idx], :].copy()
#     train_y_big_error = train_y[mark_1_idx].copy()
#     gbm_big_error = train_lgb(train_x_big_error, train_y_big_error, params_reg_big)
#     big_error_pred = gbm_big_error.predict(test_x)
#
#     train_x_small_error = train_x.loc[train_x.index[~mark_1_idx], :].copy()
#     train_y_small_error = train_y[~mark_1_idx].copy()
#     gbm_small_error = train_lgb(train_x_small_error, train_y_small_error, params_reg_small)
#     small_error_pred = gbm_small_error.predict(test_x)
#
#     return big_error_pred * prob_big + (1 - prob_big) * small_error_pred
#
#
# def pred_2layer_mid_clf(train_x, train_y, test_x):
#
#     params_clf_local = params_base.copy()
#     params_reg_local = params_base.copy()
#     params_clf_local.update(params_clf)
#     params_reg_local.update(params_reg)
#
#     params_clf_local.update(params_clf_mid_error)
#     params_reg_big = params_reg_local.copy()
#     params_reg_big.update(params_reg_mid_error_pos)
#     params_reg_small = params_reg_local.copy()
#     params_reg_small.update(params_reg_mid_error_neg)
#
#     train_y_clf = np.zeros(train_y.shape[0])
#     mark_1_idx = train_y > Y_Q_MAP[float_to_str(0.5)]
#     train_y_clf[mark_1_idx] = 1
#     gbm_clf = train_lgb(train_x, train_y_clf, params_clf_local)
#     prob_big = gbm_clf.predict(test_x)
#
#     train_x_big_error = train_x.loc[train_x.index[mark_1_idx], :].copy()
#     train_y_big_error = train_y[mark_1_idx].copy()
#     gbm_big_error = train_lgb(train_x_big_error, train_y_big_error, params_reg_big)
#     big_error_pred = gbm_big_error.predict(test_x)
#
#     train_x_small_error = train_x.loc[train_x.index[~mark_1_idx], :].copy()
#     train_y_small_error = train_y[~mark_1_idx].copy()
#     gbm_small_error = train_lgb(train_x_small_error, train_y_small_error, params_reg_small)
#     small_error_pred = gbm_small_error.predict(test_x)
#
#     return big_error_pred * prob_big + (1 - prob_big) * small_error_pred
#
#
# def pred_blending_raw_sign_clf(train_x, train_y, test_x):
#     raw_pred = lgb_raw(train_x, train_y, test_x)
#     sign_clf_pred = pred_2layer_sign_clf(train_x, train_y, test_x)
#     return (raw_pred + sign_clf_pred) / 2
#
#
# def param_search_2layer():
#     train_x, train_y, test_x = load_data_raw()
#     del test_x
#     gc.collect()
#
#     # prep for lgb
#     prep_for_lgb_single(train_x)
#     train_x_lgb = lgb_data_prep(train_x)
#
#     n_iter = 50
#
#     # 3 sets of run
#     params_clf_local = params_base.copy()
#     params_reg_local = params_base.copy()
#     params_clf_local.update(params_clf)
#     params_reg_local.update(params_reg)
#
#     # 1, clf by abs error size
#     # train_y_clf_run1 = np.zeros(train_y.shape[0])
#     # mark_1_idx = np.logical_or(train_y > Y_Q_MAP[float_to_str(0.75)], train_y < Y_Q_MAP[float_to_str(0.25)])
#     # train_y_clf_run1[mark_1_idx] = 1
#     # search_lgb_random(train_x_lgb, train_y_clf_run1, params_clf_local, label='clf_abserror', n_iter=n_iter, min_data_in_leaf_range=(200, 500), num_leaf_range=(30,50))
#     #
#     # train_x_large_error = train_x_lgb.loc[train_x_lgb.index[mark_1_idx], :].copy()
#     # train_y_large_error = train_y[mark_1_idx].copy()
#     # search_lgb_random(train_x_large_error, train_y_large_error, params_reg_local, label='lgb_large_abserror', n_iter=n_iter, min_data_in_leaf_range=(100, 300), num_leaf_range=(20, 40))
#     #
#     # train_x_small_error = train_x_lgb.loc[train_x_lgb.index[~mark_1_idx], :].copy()
#     # train_y_small_error = train_y[~mark_1_idx].copy()
#     # search_lgb_random(train_x_small_error, train_y_small_error, params_reg_local, label='lgb_small_abserror', n_iter=n_iter, min_data_in_leaf_range=(100, 300), num_leaf_range=(20, 40))
#
#     # 2, clf by sign of error
#     train_y_clf_run2 = np.zeros(train_y.shape[0])
#     mark_1_idx = train_y > 0
#     train_y_clf_run2[mark_1_idx] = 1
#     search_lgb_random(train_x_lgb, train_y_clf_run2, params_clf_local, label='clf_signerror', n_iter=n_iter, min_data_in_leaf_range=(200, 500), num_leaf_range=(30,50))
#
#     train_x_large_error = train_x_lgb.loc[train_x_lgb.index[mark_1_idx], :].copy()
#     train_y_large_error = train_y[mark_1_idx].copy()
#     search_lgb_random(train_x_large_error, train_y_large_error, params_reg_local, label='lgb_pos_signerror', n_iter=n_iter, min_data_in_leaf_range=(100, 300), num_leaf_range=(20, 40))
#
#     train_x_small_error = train_x_lgb.loc[train_x_lgb.index[~mark_1_idx], :].copy()
#     train_y_small_error = train_y[~mark_1_idx].copy()
#     search_lgb_random(train_x_small_error, train_y_small_error, params_reg_local, label='lgb_neg_signerror', n_iter=n_iter, min_data_in_leaf_range=(100, 300), num_leaf_range=(20, 40))
#
#     # # 3, clf by median of error
#     # train_y_clf_run3 = np.zeros(train_y.shape[0])
#     # mark_1_idx = train_y > Y_Q_MAP[float_to_str(0.5)]
#     # train_y_clf_run3[mark_1_idx] = 1
#     # search_lgb_random(train_x_lgb, train_y_clf_run3, params_clf_local, label='clf_miderror', n_iter=n_iter, min_data_in_leaf_range=(200, 500), num_leaf_range=(30,50))
#     #
#     # train_x_large_error = train_x_lgb.loc[train_x_lgb.index[mark_1_idx], :].copy()
#     # train_y_large_error = train_y[mark_1_idx].copy()
#     # search_lgb_random(train_x_large_error, train_y_large_error, params_reg_local, label='lgb_pos_miderror', n_iter=n_iter, min_data_in_leaf_range=(100, 300), num_leaf_range=(20, 40))
#     #
#     # train_x_small_error = train_x_lgb.loc[train_x_lgb.index[~mark_1_idx], :].copy()
#     # train_y_small_error = train_y[~mark_1_idx].copy()
#     # search_lgb_random(train_x_small_error, train_y_small_error, params_reg_local, label='lgb_neg_miderror', n_iter=n_iter, min_data_in_leaf_range=(100, 300), num_leaf_range=(20, 40))
#
#
# def cv_2layer(train_x, train_y, op_type, class_type):
#
#     params_clf_local = params_base.copy()
#     params_reg_local = params_base.copy()
#     params_clf_local.update(params_clf)
#     params_reg_local.update(params_reg)
#
#     if op_type == 'abs':
#         mark_1_idx = np.logical_or(train_y > Y_Q_MAP[float_to_str(0.75)], train_y < Y_Q_MAP[float_to_str(0.25)])
#         if class_type == 'clf':
#             params_clf_local.update(params_clf_abs_error)
#             train_y_clf = np.zeros(train_y.shape[0])
#             train_y_clf[mark_1_idx] = 1
#             train_lgb_with_val(train_x, train_y_clf, params_clf_local)
#         elif class_type == 'big':
#             params_reg_local.update(params_reg_abs_error_big)
#             train_x_large_error = train_x.loc[train_x.index[mark_1_idx], :].copy()
#             train_y_large_error = train_y[mark_1_idx].copy()
#             train_lgb_with_val(train_x_large_error, train_y_large_error, params_reg_local)
#         elif class_type == 'small':
#             params_reg_local.update(params_reg_abs_error_small)
#             train_x_large_error = train_x.loc[train_x.index[~mark_1_idx], :].copy()
#             train_y_large_error = train_y[~mark_1_idx].copy()
#             train_lgb_with_val(train_x_large_error, train_y_large_error, params_reg_local)
#
#     elif op_type == 'sign':
#         mark_1_idx = train_y > 0
#         if class_type == 'clf':
#             params_clf_local.update(params_clf_sign_error)
#             train_y_clf = np.zeros(train_y.shape[0])
#             train_y_clf[mark_1_idx] = 1
#             train_lgb_with_val(train_x, train_y_clf, params_clf_local)
#         elif class_type == 'pos':
#             params_reg_local.update(params_reg_sign_error_pos)
#             train_x_large_error = train_x.loc[train_x.index[mark_1_idx], :].copy()
#             train_y_large_error = train_y[mark_1_idx].copy()
#             train_lgb_with_val(train_x_large_error, train_y_large_error, params_reg_local)
#         elif class_type == 'neg':
#             params_reg_local.update(params_reg_sign_error_neg)
#             train_x_large_error = train_x.loc[train_x.index[~mark_1_idx], :].copy()
#             train_y_large_error = train_y[~mark_1_idx].copy()
#             train_lgb_with_val(train_x_large_error, train_y_large_error, params_reg_local)
#
#     elif op_type == 'mid':
#         mark_1_idx = train_y > Y_Q_MAP[float_to_str(0.5)]
#         if class_type == 'clf':
#             params_clf_local.update(params_clf_mid_error)
#             train_y_clf = np.zeros(train_y.shape[0])
#             train_y_clf[mark_1_idx] = 1
#             train_lgb_with_val(train_x, train_y_clf, params_clf_local)
#         elif class_type == 'pos':
#             params_reg_local.update(params_reg_mid_error_pos)
#             train_x_large_error = train_x.loc[train_x.index[mark_1_idx], :].copy()
#             train_y_large_error = train_y[mark_1_idx].copy()
#             train_lgb_with_val(train_x_large_error, train_y_large_error, params_reg_local)
#         elif class_type == 'neg':
#             params_reg_local.update(params_reg_mid_error_neg)
#             train_x_large_error = train_x.loc[train_x.index[~mark_1_idx], :].copy()
#             train_y_large_error = train_y[~mark_1_idx].copy()
#             train_lgb_with_val(train_x_large_error, train_y_large_error, params_reg_local)


def sale_month_test(month_set={'10', '11', '12'}, with_month=False):
    x_raw, y = load_train_data(prop_2016, prop_2017)
    x = lgb_data_prep(x_raw)
    if with_month:
        # x['cos_season'] = x_raw['cos_season']
        # x['sin_season'] = x_raw['sin_season']
        x['sale_month'] = x_raw['sale_month'].apply(lambda x: int(x))
        x['sale_month'] = x['sale_month'].astype('category')

    idx = x_raw['sale_month'].apply(lambda x: x in month_set)
    print('n_sample for the month: %d' % int(np.sum(idx)))
    q4_idx = x.index[idx].values
    np.random.shuffle(q4_idx)
    split_n_q4 = int(q4_idx.shape[0] / 2)
    idx_train = np.r_[x.index[~idx].values, q4_idx[:split_n_q4]]
    idx_test = q4_idx[split_n_q4:]

    train_x = x.loc[idx_train, :]
    train_y = y[idx_train]

    test_x = x.loc[idx_test, :]
    test_y = y[idx_test]

    gbm = train_lgb(train_x, train_y, get_params(p.raw_lgb_2y_1, 'reg'))

    pred_y = lgb_raw_2y_blend_pred(train_x, train_y, test_x)
    pred_y_is = lgb_raw_2y_blend_pred(train_x, train_y, train_x)

    print('IS predict miss median: %.7f' % (train_y - pred_y_is).median())
    print('IS predict miss median on month: %.7f' % (train_y[q4_idx[:split_n_q4]] - pred_y_is[q4_idx[:split_n_q4]]).median())
    print('OS predict miss median: %.7f' % (test_y - pred_y).median())

    return gbm


def sale_month_test_by_year(month_set={'01'}):
    """See the relationship between 2016 mis-predict and 2017 mis-predict"""
    x_raw, y = load_train_data(prop_2016, prop_2017)
    x = lgb_data_prep(x_raw)

    idx = x_raw['sale_month'].apply(lambda x: x in month_set)
    idx_2016 = np.logical_and(idx, x_raw['data_year'] == 2016)
    idx_2017 = np.logical_and(idx, x_raw['data_year'] == 2017)

    pred_mon_2016 = pred_lgb_blend(x.loc[x.index[idx_2016], :])
    pred_mon_2017 = pred_lgb_blend(x.loc[x.index[idx_2017], :])

    print('predict miss median 2016: %.7f' % (y[idx_2016] - pred_mon_2016).median())
    print('predict miss median 2017: %.7f' % (y[idx_2017] - pred_mon_2017).median())


def sale_month_test_by_year_catboost(mon=1):
    """See the relationship between 2016 mis-predict and 2017 mis-predict"""
    x_train = pkl.load(open('only_catboost_x_train_v2.pkl', 'rb'))
    index_all = np.array(range(x_train.shape[0]))
    x_train.index = index_all

    y_train = pkl.load(open('only_catboost_y_train_v2.pkl', 'rb'))
    y_train.index = index_all

    y_pred = pkl.load(open('only_catboost_y_pred_train_v2.pkl', 'rb'))
    y_pred = pd.Series(y_pred, index=index_all)

    idx_mon = x_train['transaction_month'] == mon
    idx_mon_2016 = np.logical_and(idx_mon, x_train['transaction_year'] == 2016)
    idx_mon_2017 = np.logical_and(idx_mon, x_train['transaction_year'] == 2017)

    print('processing month %d' % mon)
    print('predict miss median mon: %.7f' % (y_train[idx_mon] - y_pred[idx_mon]).median())
    print('predict miss median 2016: %.7f' % (y_train[idx_mon_2016] - y_pred[idx_mon_2016]).median())
    print('predict miss median 2017: %.7f' % (y_train[idx_mon_2017] - y_pred[idx_mon_2017]).median())


def sale_month_test_by_year_catboost_batch():
    for mon in range(1, 10):
        sale_month_test_by_year_catboost(mon)


def train_lgb_with_val_one_month(mon, year, param, size_down=False):
    x_raw, y = rm_outlier(train_x, train_y)
    x_step1 = lgb_data_prep(x_raw, p.class3_new_features, p.class3_rm_features)
    x_step2 = lgb_data_prep(x_raw, keep_only_feature=p.step2_keep_only_feature)

    if not os.path.exists('raw_gbm_blending.pkl'):
        gbms = train_gbms_blend(x_step1, y, p.raw_lgb_2y)
        pkl.dump(gbms, open('raw_gbm_blending.pkl', 'wb'))
    else:
        gbms = pkl.load(open('raw_gbm_blending.pkl', 'rb'))

    idx = x_raw.index[np.logical_and(x_raw['sale_month'].apply(lambda x: x == mon), x_raw['data_year'] == year)].values
    if size_down:
        np.random.shuffle(idx)
        idx = idx[:]
    pred_raw = pred_lgb_blend(x_step1.loc[idx, :], gbms)
    error = y[idx] - pred_raw

    train_lgb_with_val(x_step2.loc[idx, :], error, get_params(param, 'reg'))


def month_sample_count():
    x_raw, y = load_train_data(prop_2016, prop_2017)
    sale_month = x_raw['sale_month']
    data_year = x_raw['data_year']
    for month in sorted(list(sale_month.unique())):
        for year in sorted(list(data_year.unique())):
            idx = np.logical_and(sale_month == month, data_year == year)
            print('n sample for month %s, year %d: %d' % (month, year, int(np.sum(idx))))


def train_mon_2step(mon_set):
    """first train with whole data.
       then train with 2016 month data on errors of that month set and predict for 2017"""
    print('processing month set %s' % str(mon_set))
    x_raw, y = rm_outlier(train_x, train_y)
    x_step1 = lgb_data_prep(x_raw, p.class3_new_features, p.class3_rm_features)
    x_step2 = lgb_data_prep(x_raw, keep_only_feature=p.step2_keep_only_feature)

    idx_2016 = x_raw.index[np.logical_and(x_raw['sale_month'].apply(lambda x: x in mon_set), x_raw['data_year'] == 2016)]
    idx_2017 = x_raw.index[np.logical_and(x_raw['sale_month'].apply(lambda x: x in mon_set), x_raw['data_year'] == 2017)]

    # raw_pred_2016 = pred_lgb_blend(x_step1.loc[x_raw.index[idx_2016], :])
    # raw_pred_2017 = pred_lgb_blend(x_step1.loc[x_raw.index[idx_2017], :])

    # error_2016 = train_y[idx_2016] - raw_pred_2016
    pred_step1 = pd.Series(pkl.load(open('final_pred/pred_step1_train.pkl', 'rb')), index=x_raw.index)
    error_2016 = y[idx_2016] - pred_step1[idx_2016]
    error_2017 = y[idx_2017] - pred_step1[idx_2017]

    # train on 2016 error and predict for 2017 error
    error_gbms  = train_gbms_blend(x_step2.loc[idx_2016, :], error_2016, p.lgb_month)
    error_2016_pred = pred_lgb_blend(x_step2.loc[idx_2016, :], error_gbms)
    error_2017_pred = pred_lgb_blend(x_step2.loc[idx_2017, :], error_gbms)
    error_2017_1 = train_y[idx_2017] - (pred_step1[idx_2017] + error_2017_pred)
    error_2017_2 = train_y[idx_2017] - (pred_step1[idx_2017] + error_2017_pred / 2)

    print('2016 step1 error median: %.7f' % np.median(error_2016))
    print('error_2016 pred median: %.7f' % np.median(error_2016_pred))
    print('IS(2016) predict miss median with full error: %.7f' % (error_2016 - error_2016_pred).median())
    print('IS(2016) predict miss median with half error: %.7f' % (error_2016 - error_2016_pred / 2).median())

    print('2017 step1 error median: %.7f' % error_2017.median())
    print('error_2017 pred median: %.7f' % np.median(error_2017_pred))
    print('OS(2017) predict miss median with full error adj: %.7f' % error_2017_1.median())
    print('OS(2017) predict miss median:with half error adj %.7f' % error_2017_2.median())


def pred_train_mon_2step(rm_outlier_flag=False):
    mon_set = {'10', '11', '12'}
    x_raw, y = load_train_data(prop_2016, prop_2017)
    if rm_outlier_flag:
        x_raw, y = rm_outlier(x_raw, y)

    x = lgb_data_prep(x_raw)
    prop_2016_lgb = lgb_data_prep(prop_2016)
    idx = x_raw['sale_month'].apply(lambda x: x in mon_set).values
    raw_pred_train = pred_lgb_blend(x.loc[x.index[idx], :])
    error_train = y[idx] - raw_pred_train
    error_gbm = train_lgb(x.loc[x.index[idx]], error_train, get_params(p.lgb_month_1, 'reg'))

    raw_pred_submit = pred_lgb_blend(prop_2016_lgb)
    error_pred_submit = error_gbm.predict(prop_2016_lgb)

    return raw_pred_submit + error_pred_submit


def pred_2016_to_2017():
    mon_sets = ({'01'}, {'02'}, {'03'}, {'04'}, {'05'}, {'06'}, {'07'}, {'08'}, {'09'})
    # x_raw, y = capfloor_outlier(train_x, train_y)
    x_raw, y = rm_outlier(train_x, train_y)
    # x_raw, y = train_x, train_y
    x = lgb_data_prep(x_raw, p.class3_new_features, p.class3_rm_features)
    x_step2 = lgb_data_prep(x_raw, keep_only_feature=p.step3_keep_only_feature)
    idx_2016 = (x_raw['data_year'] == 2016).values
    idx_2017 = (x_raw['data_year'] == 2017).values
    # get all year_error data
    # gbms_step1 = train_gbms_blend(x, y, p.raw_lgb_2y)
    # raw_pred = pred_lgb_blend(x, gbms_step1)
    raw_pred = pkl.load(open('final_pred/pred_step1_train.pkl', 'rb'))
    raw_pred = pd.Series(raw_pred, index=x.index)

    # train_test_split
    train_2017_error_idx, val_2017_error_idx = train_test_split(x_raw.index[idx_2017].values, stratify=x_raw.loc[x_raw.index[idx_2017], 'sale_month'], test_size=0.3, random_state=42)

    pred_year = pd.Series(np.zeros(np.sum(idx_2017)), index=x_raw.index[idx_2017])  # error after month training for 2017
    pred_year_2017_month_all = {}
    for mon_set in mon_sets:
        print('processing month set: %s' % str(mon_set))
        idx_2016_month = x_step2.index[np.logical_and(x_raw['sale_month'].apply(lambda x: x in mon_set), idx_2016)]
        idx_2017_month = x_step2.index[np.logical_and(x_raw['sale_month'].apply(lambda x: x in mon_set), idx_2017)]
        error_2016 = train_y[idx_2016_month] - raw_pred[idx_2016_month]
        error_gbm = train_lgb(x_step2.loc[idx_2016_month, :], error_2016, get_params(p.lgb_month_1, 'reg'))
        print('before month train error median: %.7f' % np.median(train_y[idx_2017_month] - raw_pred[idx_2017_month]))
        error_2017_pred = error_gbm.predict(x_step2.loc[idx_2017_month, :])
        pred_year_2017_month = raw_pred[idx_2017_month] + error_2017_pred
        pred_year_2017_month_all[str(mon_set)] = pred_year_2017_month
        print('after month train error median: %.7f' % np.median(train_y[idx_2017_month] - pred_year_2017_month))
        # validation set error median after month train
        val_idx_month = val_2017_error_idx[x_raw.loc[val_2017_error_idx, 'sale_month'].apply(lambda x: x in mon_set)]
        print('after month train validation error median: %.7f' % np.median(train_y[val_idx_month] - pred_year_2017_month[val_idx_month]))
        pred_year[idx_2017_month] = pred_year_2017_month

    # error_year = train_2017_y - pred_year
    # # pkl.dump(error_year, open('error_after_month_train_2017.pkl', 'wb'))
    #
    # # adj for year
    # x_year_error = lgb_data_prep(x_raw, keep_only_feature=('2y_diff_dollar_taxvalue_total', '2y_diff_dollar_taxvalue_land', '2y_diff_dollar_taxvalue_structure'))
    # error_year_train = error_year[train_2017_error_idx]
    # x_year_error_train = x_year_error.loc[train_2017_error_idx, :]
    # gbm_year_error = train_lgb(x_year_error_train, error_year_train, get_params(p.lgb_year_1, 'reg'))
    # for mon_set in mon_sets:
    #     val_idx_month = val_2017_error_idx[x_raw.loc[val_2017_error_idx, 'sale_month'].apply(lambda x: x in mon_set)]
    #     val_month_year_adj = gbm_year_error.predict(x_year_error.loc[val_idx_month, :])
    #     val_month_pred_error = train_y[val_idx_month] - pred_year_2017_month_all[str(mon_set)][val_idx_month] - val_month_year_adj
    #     print('%s after year train validation error median: %.7f' % (str(mon_set), np.median(val_month_pred_error)))


def train_mon_2step_batch_run():
    for mon_set in ({'01'}, {'02'}, {'03'}, {'04'}, {'05'}, {'06'}, {'07'}, {'08'}, {'09'}):
        train_mon_2step(mon_set)


def train_gbms_blend(train_x, train_y, params):
    # print('re-train lgb blending')
    gbms = []
    for param in params:
        gbms.append(train_lgb(train_x, train_y, get_params(param, 'reg')))
    return gbms


def pred_lgb_blend(test_x, gbms=None):
    if not gbms:
        print('load gbm from local')
        gbms = pkl.load(open('raw_lgb_blending.pkl', 'rb'))
    raw_pred = []
    for gbm in gbms:
        raw_pred.append(gbm.predict(test_x))
    return np.array(raw_pred).mean(axis=0)


# def pred_fe3_lgb_blend(test_x):
#     gbms = pkl.load(open('fe3_lgb_blending.pkl', 'rb'))
#     raw_pred = []
#     for gbm in gbms:
#         raw_pred.append(gbm.predict(test_x))
#     return np.array(raw_pred).mean(axis=0)
#
#
# def pred_fe3_lgb_blend_submit(prop_data):
#     new_features = load_feature_list('2y_raw_lgb')
#     for f in new_features:
#         if f not in prop_data.columns:
#             feature_factory(f, prop_data)
#     prop_data_use = lgb_data_prep(prop_data, new_features)
#     gbms = pkl.load(open('fe3_lgb_blending.pkl', 'rb'))
#     raw_pred = []
#     for gbm in gbms:
#         raw_pred.append(gbm.predict(prop_data_use))
#     return np.array(raw_pred).mean(axis=0)


# def model_2layer(train_x, train_y, test_x):
#     """input train_y is expected to be original log error"""
#     # for each x clf large / small error
#     train_x_error_clf, train_y_error_clf, test_x_error_clf = error_clf_data_prep(test_x, train_x, train_y)
#     gbm_clf = error_clf_train(train_x_error_clf, train_y_error_clf)
#     error_clf_prob = gbm_clf.predict(test_x_error_clf)
#
#     # predict x with large error model
#     train_x_error_big, test_x_error_big = big_error_data_prep(test_x, train_x)
#     gbm_error_big = big_error_train(train_x_error_big, train_y)
#     pred_error_big = gbm_error_big.predict(test_x_error_big)
#
#     # predict x with small error model
#     train_x_error_small, test_x_error_small = small_error_data_perp(test_x, train_x)
#     gbm_error_small = big_error_train(train_x_error_small, train_y)
#     pred_error_small = gbm_error_big.predict(test_x_error_small)
#
#     # final prediction
#     pred = pred_error_big * error_clf_prob + pred_error_small * (1 - error_clf_prob)
#     return pred
#
#
# # -------------------------------------------------------- SVR ------------------------------------------------------------
# def svr_data_prep(train_x):
#     feature_float = ['area_living_finished_calc',
#                     'latitude', 'longitude', 'area_lot', 'area_pool',
#                     'dollar_taxvalue_structure', 'dollar_taxvalue_total', 'dollar_taxvalue_land', 'dollar_tax']
#     feature_int = ['num_room', 'year_built', 'num_bathroom_assessor', 'num_bedroom', 'num_fullbath']
#     keep_feature = feature_float + feature_int
#     svr_train_data = train_x[keep_feature].copy()
#     for f in feature_float:
#         svr_train_data[f].fillna(svr_train_data[f].mean(), inplace=True)
#     for f in feature_int:
#         svr_train_data[f].fillna(svr_train_data[f].mode()[0], inplace=True)
#
#     svr_train_data['dollar_taxvalue_structure_land_diff'] = svr_train_data['dollar_taxvalue_structure'] - svr_train_data['dollar_taxvalue_land']
#     svr_train_data['dollar_taxvalue_structure_land_absdiff'] = np.abs(svr_train_data['dollar_taxvalue_structure'] - svr_train_data['dollar_taxvalue_land'])
#     svr_train_data['dollar_taxvalue_structure_land_diff_norm'] = (svr_train_data['dollar_taxvalue_structure'] - svr_train_data['dollar_taxvalue_land']) / svr_train_data['dollar_taxvalue_total']
#     svr_train_data['dollar_taxvalue_structure_land_absdiff_norm'] = np.abs(svr_train_data['dollar_taxvalue_structure'] - svr_train_data['dollar_taxvalue_land']) / svr_train_data['dollar_taxvalue_total']
#     svr_train_data['dollar_taxvalue_structure_total_ratio'] = svr_train_data['dollar_taxvalue_structure'] / svr_train_data['dollar_taxvalue_total']
#     svr_train_data['dollar_taxvalue_total_dollar_tax_ratio'] = svr_train_data['dollar_taxvalue_total'] / svr_train_data['dollar_tax']
#     svr_train_data['living_area_proportion'] = svr_train_data['area_living_finished_calc'] / svr_train_data['area_lot']
#
#     for v in ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total'):
#         for a in ('area_lot', 'area_living_finished_calc'):
#             col_name = v + '_per_' + a
#             svr_train_data[col_name] = svr_train_data[v] / svr_train_data[a]
#
#     # create high cardinality var mapping
#     for cat_var in ['str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
#                 'raw_block', 'raw_census', 'block', 'census']:
#         for num_var in ['area_living_finished_calc',
#                     'latitude', 'longitude', 'area_lot', 'area_pool',
#                     'dollar_taxvalue_structure', 'dollar_taxvalue_total', 'dollar_taxvalue_land', 'dollar_tax']:
#             var_name = num_var + '_group_mean_by_' + cat_var
#             cat_var_col = train_x[cat_var].copy()
#             cat_var_col.fillna(cat_var_col.mode()[0], inplace=True)
#             svr_train_data[var_name] = cat_var_col.map(svr_train_data[num_var].groupby(cat_var_col).mean())
#
#     # normalize predictors for SVM
#     scr_train_data = pd.DataFrame(normalize(svr_train_data), columns=svr_train_data.columns, index=svr_train_data.index)
#     return scr_train_data
#
#
# def svr_grid_search(x, y):
#     Cs = [0.001, 0.01, 0.1, 1, 10]
#     epsilons = [0.0001, 0.001, 0.01, 0.1, 1]
#     param_grid = {'C': Cs, 'epsilon' : epsilons}
#     grid_search = GridSearchCV(LinearSVR(random_state=42), param_grid, cv=5, scoring='neg_mean_absolute_error')
#     grid_search.fit(x, y)
#     return grid_search
#
#
# def svr_random_search(x, y):
#     param_dist = {'C': np.random.uniform(0.1, 2, 20).tolist(),
#                   'epsilon': np.random.uniform(0.001, 0.015, 20).tolist()}
#     random_search = RandomizedSearchCV(LinearSVR(random_state=42), param_dist, cv=5, scoring='neg_mean_absolute_error', n_iter=30)
#     random_search.fit(x, y)
#     return random_search
#
#
# def svr_train(x, y):
#     svr = LinearSVR(random_state=42, C=1, epsilon=0.0025)
#     svr.fit(x, y)
#     return svr
#
#
# def svr_pred(x, y, x_test):
#     svr = svr_train(x, y)
#     return svr.predict(x_test)
#
#
# def svr_cv(x, y):
#     return cv_meta_model(x, y, None, svr_pred, outlier_thresh=0.001, outlier_handling=None)
#
#
# def svr_lgb_stack_pred(train_x, train_y, test_x):
#     svr_train_x = svr_data_prep(train_x)
#     svr_test_x = svr_data_prep(test_x)
#     svr_pred_y = svr_pred(svr_train_x, train_y, svr_test_x)
#     svr_residual = train_y - svr_pred(svr_train_x, train_y, svr_train_x)
#
#     # prep for lgb categorical variables should be applied outside of this function call
#     lgb_train_x = lgb_data_prep(train_x)
#     lgb_test_x = lgb_data_prep(test_x)
#     gbm = train_lgb(lgb_train_x, svr_residual)
#     lgb_pred = gbm.predict(lgb_test_x)
#     return lgb_pred + svr_pred_y
#
#
# def svr_lgb_blend_pred(train_x, train_y, test_x):
#     svr_train_x = svr_data_prep(train_x)
#     svr_test_x = svr_data_prep(test_x)
#     svr = svr_train(svr_train_x, train_y)
#     svr_train_y = svr.predict(svr_train_x)
#     svr_test_y = svr.predict(svr_test_x)
#
#     lgb_train_x = lgb_data_prep(train_x)
#     lgb_test_x = lgb_data_prep(test_x)
#     gbm = train_lgb(lgb_train_x, train_y)
#     lgb_train_y = gbm.predict(lgb_train_x)
#     lgb_test_y = gbm.predict(lgb_test_x)
#
#     # # fit linear reg
#     # lr = LinearRegression()
#     # lr.fit(np.array([svr_train_y, lgb_train_y]).T, train_y)
#     #
#     # svr_coef, lgb_coef = lr.coef_
#     # print('intercept: %.6f' % lr.intercept_)
#     # print('svr_coef: %.6f' % svr_coef)
#     # print('lgb_coef: %.6f' % lgb_coef)
#     #
#     # # use liear reg to predict
#     # return lr.predict(np.array([svr_test_y, lgb_test_y]).T)
#
#     return (svr_test_y + lgb_test_y) / 2