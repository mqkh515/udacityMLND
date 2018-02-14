import pandas as pd
import numpy as np
import data_prep
import lightgbm as lgb


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')


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
    params = {
        'boosting_type': 'gbdt',
        'feature_fraction': 0.95,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': 0,
        'lambda_l2': 0,
        'objective': 'regression_l1',
        'metric': {'l1'}
        }
    lgb_train = lgb.Dataset(train_x, train_y)
    metric = list(params['metric'])[0]
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
        params.update(rand_params)
        eval_hist = lgb.cv(params, lgb_train, stratified=False, num_boost_round=12000, early_stopping_rounds=100)
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
    res_df.to_csv('param_search/lgb_random_%s.csv' % label, index=False)


def param_search_runner():
    train_x = lgb_model_prep(data_prep.train_x)
    param_search_lgb_random(train_x, data_prep.train_y, 'raw')


if __name__ == '__main__':
    param_search_runner()




