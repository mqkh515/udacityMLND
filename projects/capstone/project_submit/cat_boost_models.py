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
        if col in ['num_bathroom_assessor', 'code_county', 'area_living_type_12', 'area_firstfloor_assessor']:
            continue
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


def param_search(train_x, train_y,  n_iter=50, depth=(5, 8), l2_leaf_reg=(1, 10)):

    def rand_depth():
        return np.random.randint(depth[0], depth[1])

    def rand_learning_rate():
        """learning rate use log linear spacing"""
        return np.random.uniform(0.015, 0.045)

    def rand_l2_leaf_reg():
        return np.random.randint(l2_leaf_reg[0], l2_leaf_reg[1])

    train_x['sale_month_derive'] = train_x['sale_month'].apply(lambda x: x if x not in {10, 11, 12} else 10)
    train_x, train_y = data_prep.rm_outlier(train_x, train_y)
    train_x, cat_inds = cat_boost_model_prep(train_x, ['sale_month_derive'], [False])

    np.random.seed(7)

    def write_to_file(line):
        f = open('param_search/catboost_random.txt', 'a')
        f.write(line + '\n')
        f.close()

    headers = ','.join(['MAE_avg', 'MAE_stddev', 'n_iter', 'learning_rate', 'depth', 'l2_leaf_reg'])
    write_to_file(headers)

    def do_cv(learning_rate, depth, l2_leaf_reg):
        param = {'iterations': 3000,
                 'od_type': 'Iter',
                 'od_wait': 50,
                 'learning_rate':learning_rate ,
                 'depth': depth,
                 'l2_leaf_reg': l2_leaf_reg,
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
        line = '%.7f,%.7f,%.0f,%.6f,%.0f,%.0f' % tuple(res_list)
        write_to_file(line)
        return res_list

    res = [do_cv(0.03, 6, 3)]  # cv res from default param
    for i in range(1, n_iter + 1):
        res.append(do_cv(rand_learning_rate(), rand_depth(), rand_l2_leaf_reg()))
        print('finished %d / %d' % (i, n_iter))
    res_df = pd.DataFrame(res, columns=['MAE_avg', 'MAE_stddev', 'n_iter', 'learning_rate', 'depth', 'l2_leaf_reg'])
    res_df.sort_values('MAE_avg', inplace=True)
    res_df.to_csv('param_search/catboost_random.csv', index=False)
