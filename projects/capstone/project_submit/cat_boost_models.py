from catboost import CatBoostRegressor
import catboost
import data_prep
import numpy as np
import pandas as pd


feature_info = data_prep.feature_info


def cat_boost_model_prep(data, added_features=(), added_features_cat_flag=()):
    """mark categorical column indexes, and set NaN values to -999 for numerical values as demonstrated in catboost examples."""
    features_all = list(feature_info.index.values)
    features_list = list(added_features)
    cat_flag_list = list(added_features_cat_flag)
    for col in features_all:
        if feature_info.loc[col, 'type'] in {'cat', 'num'}:
            features_list.append(col)
        else:
            continue
        if feature_info.loc[col, 'type'] == 'cat':
            cat_flag_list.append(True)
        else:
            cat_flag_list.append(False)

    data = data[features_list]
    data.fillna(-999, inplace=True)
    cat_inds = np.array(range(len(cat_flag_list)))[cat_flag_list].tolist()
    print('cat features: ' + ', '.join(np.array(features_list)[cat_inds]))
    return data, cat_inds


def cat_boost_obj_gen(param, seed):
    return catboost.CatBoostRegressor(iterations=param['iterations'],
                                      learning_rate=param['learning_rate'],
                                      depth=param['depth'],
                                      l2_leaf_reg=param['l2_leaf_reg'],
                                      loss_function='MAE',
                                      eval_metric='MAE',
                                      random_seed=seed)


def cat_boost_param_search(train_x, train_y,  n_iter=100, depth=(5, 11), l2_leaf_reg=(2, 10)):

    def rand_depth():
        return np.random.randint(depth[0], depth[1])

    def rand_learning_rate():
        """learning rate use log linear spacing"""
        return np.random.uniform(1, 8) * 0.01

    def rand_l2_leaf_reg():
        return np.random.randint(l2_leaf_reg[0], l2_leaf_reg[1])

    train_x['sale_month_derive'] = train_x['sale_month'].apply(lambda x: x if x not in {10, 11, 12} else 10)
    train_x, cat_inds = cat_boost_model_prep(train_x, ['sale_month_derive'], [False])

    np.random.seed(7)

    res = []
    for i in range(1, n_iter + 1):
        param = {'iterations': 2000,
                 'od_type': 'Iter',
                 'od_wait': 20,
                 'learning_rate': rand_learning_rate(),
                 'depth': rand_depth(),
                 'l2_leaf_reg': rand_l2_leaf_reg(),
                 'loss_function': 'MAE',
                 'eval_metric': 'MAE',
                 }
        eval_hist = catboost.cv(param, catboost.Pool(train_x, train_y, cat_inds), 5)
        res_list = [eval_hist['MAE_test_avg'][-1],
                    eval_hist['MAE_test_stddev'][-1],
                    len(eval_hist['MAE_test_avg']),
                    param['learning_rate'],
                    param['depth'],
                    param['l2_leaf_reg'],
                    ]
        res.append(res_list)
        print('finished %d / %d' % (i, n_iter))
    res_df = pd.DataFrame(res, columns=['MAE_avg', 'MAE_stddev', 'n_iter', 'learning_rate', 'depth', 'l2_leaf_reg'])
    res_df.sort_values('MAE_avg', inplace=True)
    res_df.to_csv('param_search/catboost_random.csv', index=False)


def cat_boost_test():
    x = pd.DataFrame(np.random.randn(1000, 3))
    y = x[0] + 1 + x[1] + 2 + x[2] * 3 + np.random.randn(1000) / 2
    res = catboost.cv({}, catboost.Pool(x, y), 5)