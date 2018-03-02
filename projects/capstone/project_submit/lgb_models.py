import pandas as pd
import numpy as np
import data_prep
import lightgbm as lgb
import params


feature_info = data_prep.feature_info


PARAMS = {
    'boosting_type': 'gbdt',
    'feature_fraction': 0.95,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': 0,
    'lambda_l2': 0,
    'objective': 'regression_l1',
    'metric': {'l1'}
}


def lgb_model_prep(data, new_features=tuple(), rm_features=tuple(), keep_only_feature=()):
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

    for col in ['num_bathroom_assessor', 'code_county', 'area_living_type_12', 'area_firstfloor_assessor']:
        if col in keep_feature:
            keep_feature.remove(col)

    if len(keep_only_feature) > 0:
        keep_feature = list(keep_only_feature)

    return data[keep_feature]


def param_search_lgb_random(train_x, train_y, label='', n_iter=100, min_data_in_leaf_range=(100, 600), num_leaf_range=(30, 80)):
    """random param search. only search for min_data_in_leaf, num_leaf_range and learning rate, L2 norm is set to zero as min_data_in_leaf acts on the same target as L2 coef."""
    np.random.seed(7)
    p = PARAMS.copy()
    lgb_train = lgb.Dataset(train_x, train_y)
    metric = list(p['metric'])[0]
    columns = ['%s-mean' % metric, '%s-stdv' % metric, 'n_rounds', 'num_leaves', 'min_data_in_leaf', 'learning_rate']

    def rand_min_data_in_leaf():
        return np.random.randint(min_data_in_leaf_range[0], min_data_in_leaf_range[1])

    def rand_learning_rate():
        """learning rate use log linear spacing"""
        return np.random.uniform(1, 3)

    def rand_num_leaf():
        return np.random.randint(num_leaf_range[0], num_leaf_range[1])

    res = []
    for i in range(1, n_iter + 1):
        rand_params = {'num_leaves': rand_num_leaf(),
                       'min_data_in_leaf': rand_min_data_in_leaf(),
                       'learning_rate': 0.1 ** rand_learning_rate(),
                       }
        p.update(rand_params)
        eval_hist = lgb.cv(p, lgb_train, stratified=False, num_boost_round=12000, early_stopping_rounds=100)
        res_list = [eval_hist['%s-mean' % metric][-1],
                    eval_hist['%s-stdv' % metric][-1],
                    len(eval_hist['%s-mean' % metric]),
                    rand_params['num_leaves'],
                    rand_params['min_data_in_leaf'],
                    rand_params['learning_rate'],
                    ]
        res.append(res_list)
        print('finished %d / %d' % (i, n_iter))
    res_df = pd.DataFrame(res, columns=columns)
    res_df.sort_values('%s-mean' % metric, inplace=True)
    res_df.to_csv('param_search/lgb_random_%s.csv' % label, index=False)


def train_lgb(train_x, train_y, params_inp):
    lgb_train = lgb.Dataset(train_x, train_y)
    p = params_inp.copy()
    num_boost_round = p.pop('num_boosting_rounds')
    gbm = lgb.train(p, lgb_train, num_boost_round=num_boost_round)
    return gbm


def param_search_raw():
    """corresponds to ModelLGBRaw"""
    train_x = lgb_model_prep(data_prep.train_x)
    param_search_lgb_random(train_x, data_prep.train_y, 'raw')


def param_search_one_step():
    """corresponds to ModelLGBOneStep, with sale_month included, with class3_FE and outlier rm"""
    train_x = data_prep.train_x.copy()
    train_x['sale_month_derive'] = train_x['sale_month'].apply(lambda x: x if x < 10 else 10)
    new_features = params.class3_new_features + ['sale_month_derive']
    train_x = lgb_model_prep(train_x, new_features, params.class3_rm_features)
    train_x, train_y = data_prep.rm_outlier(train_x, data_prep.train_y)
    param_search_lgb_random(train_x, train_y, '1step')


def param_search_step2(n_iter=100, min_data_in_leaf_range=(30, 100), num_leaf_range=(5, 20)):
    """corresponds to step 2 of ModelLGBTwoStep, with no sale month included in step 1, and selected features only in step 2."""
    train_x, train_y = data_prep.rm_outlier(data_prep.train_x, data_prep.train_y)

    x_step1 = lgb_model_prep(train_x, params.class3_new_features, params.class3_rm_features)
    params_step1 = PARAMS.copy()
    params_step1.update(params.lgb_step1_1)
    gbm = train_lgb(x_step1, train_y, params_step1)
    pred_step1 = gbm.predict(x_step1)
    error_step1 = train_y - pred_step1  # pd.Series

    def rand_min_data_in_leaf():
        return np.random.randint(min_data_in_leaf_range[0], min_data_in_leaf_range[1])

    def rand_learning_rate():
        return np.random.uniform(2, 3)

    def rand_num_leaf():
        return np.random.randint(num_leaf_range[0], num_leaf_range[1])

    x_step2 = lgb_model_prep(train_x, keep_only_feature=params.step2_keep_only_feature)
    np.random.seed(7)
    p = PARAMS.copy()
    metric = list(p['metric'])[0]
    columns = ['%s-mean' % metric, '%s-stdv' % metric, 'n_rounds-mean', 'n_rounds_stdv', 'num_leaves', 'min_data_in_leaf', 'learning_rate']

    res = []
    for i in range(1, n_iter + 1):
        rand_params = {'num_leaves': rand_num_leaf(),
                       'min_data_in_leaf': rand_min_data_in_leaf(),
                       'learning_rate': 0.1 ** rand_learning_rate()
                       }
        p.update(rand_params)
        cv_hist = []
        n_rounds = []
        for mon in range(1, 10):
            use_idx = np.array(train_x.index[train_x['sale_month'] == mon])
            np.random.shuffle(use_idx)
            use_idx = use_idx[:9000]  # always use only 9000 samples to match the sample size of 2016-10,11,12
            pred_error = error_step1[use_idx]
            train_x_step2_local = x_step2.loc[use_idx, :]
            lgb_train = lgb.Dataset(train_x_step2_local, pred_error)
            eval_hist = lgb.cv(p, lgb_train, stratified=False, num_boost_round=5000, early_stopping_rounds=50)
            cv_hist.append([eval_hist['%s-mean' % metric][-1], eval_hist['%s-stdv' % metric][-1]])
            n_rounds.append(len(eval_hist['%s-mean' % metric]))
        m_mean, m_stdv = np.array(cv_hist).mean(axis=0)
        n_rounds_mean = np.mean(np.array(n_rounds))
        n_rounds_stdv = np.std(np.array(n_rounds))
        res.append([m_mean, m_stdv, n_rounds_mean, n_rounds_stdv, rand_params['num_leaves'], rand_params['min_data_in_leaf'], rand_params['learning_rate']])
        print('finished %d / %d' % (i, n_iter))
    res_df = pd.DataFrame(res, columns=columns)
    res_df.sort_values('%s-mean' % metric, inplace=True)
    res_df.to_csv('param_search/lgb_random_step2.csv', index=False)














