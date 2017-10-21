import pandas as pd
import numpy as np
import zillow_1 as z
import lightgbm as lgb
import params_cache as p
import pickle as pkl
import final_pred_script
import only_catboost


def param_search_batch():
    new_features = z.load_feature_list('2y_raw_lgb')
    train_x, train_y = z.load_train_data(z.prop_2016, z.prop_2017, new_features)
    train_x_lgb = z.lgb_data_prep(train_x, new_features)

    n_iter = 100
    params_reg = z.params_base.copy()
    params_reg.update(z.params_reg)
    # params_clf = z.params_base.copy()
    # params_clf.update(z.params_clf)

    # raw lgb
    z.search_lgb_random(train_x_lgb, train_y, params_reg, 'raw_lgb', n_iter)

    # # clf
    # train_y_local = np.zeros(train_y.shape)
    # train_y_local[train_y > 0] = 1
    # z.search_lgb_random(train_x_lgb, train_y_local, params_clf, 'sign_error_clf', n_iter)
    #
    # # pos error
    # mark_idx = train_y > 0
    # train_x_local = train_x_lgb.loc[train_x_lgb.index[mark_idx], :]
    # train_y_local = train_y.loc[mark_idx]
    # z.search_lgb_random(train_x_local, train_y_local, params_reg, 'sign_error_pos', n_iter)
    #
    # # neg error
    # mark_idx = train_y < 0
    # train_x_local = train_x_lgb.loc[train_x_lgb.index[mark_idx], :]
    # train_y_local = train_y.loc[mark_idx]
    # z.search_lgb_random(train_x_local, train_y_local, params_reg, 'sign_error_neg', n_iter)

    # --------------------------------BO search------------------------------
    # # raw lgb
    # z.search_lgb_bo(train_x_lgb, train_y, params_reg, 'raw_lgb', n_iter)
    #
    # # clf
    # train_y_local = np.zeros(train_y.shape)
    # train_y_local[train_y > 0] = 1
    # z.search_lgb_bo(train_x_lgb, train_y_local, params_clf, 'sign_error_clf', n_iter, do_clf=True)
    #
    # # pos error
    # mark_idx = train_y > 0
    # train_x_local = train_x_lgb.loc[train_x_lgb.index[mark_idx], :]
    # train_y_local = train_y.loc[mark_idx]
    # z.search_lgb_bo(train_x_local, train_y_local, params_reg, 'sign_error_pos', n_iter)
    #
    # # neg error
    # mark_idx = train_y < 0
    # train_x_local = train_x_lgb.loc[train_x_lgb.index[mark_idx], :]
    # train_y_local = train_y.loc[mark_idx]
    # z.search_lgb_bo(train_x_local, train_y_local, params_reg, 'sign_error_neg', n_iter)


def param_search_batch_with_outlier():
    new_features = z.load_feature_list('2y_raw_lgb')
    train_x, train_y = z.load_train_data(z.prop_2016, z.prop_2017, new_features)
    train_x_lgb = z.lgb_data_prep(train_x, new_features)

    n_iter = 150
    params_reg = z.params_base.copy()
    params_reg.update(z.params_reg)

    # raw lgb
    z.search_lgb_random(train_x_lgb, train_y, params_reg, 'lgb_fe3', n_iter, with_rm_outlier=True)


def param_search_raw_lgb_final():
    x_raw, y = z.rm_outlier(z.train_x, z.train_y)
    x = z.lgb_data_prep(x_raw, p.class3_new_features, p.class3_rm_features)

    n_iter = 100
    params_reg = z.params_base.copy()
    params_reg.update(z.params_reg)

    # raw lgb
    z.search_lgb_random(x, y, params_reg, 'lgb_raw_final', n_iter)


def param_search_batch_one_mon(keep_size=('all',)):
    x_raw, y = z.rm_outlier(z.train_x, z.train_y)
    x_step1 = z.lgb_data_prep(x_raw, p.class3_new_features, p.class3_rm_features)
    x_step2 = z.lgb_data_prep(x_raw, keep_only_feature=p.step2_keep_only_feature)

    n_iter = 10
    params = z.params_base.copy()
    params.update(z.params_reg)

    if 'num_boosting_rounds' in params:
        params.pop('num_boosting_rounds')

    metric = list(params['metric'])[0]
    min_data_in_leaf_range = (30, 100)
    num_leaf_range = (5, 20)

    def rand_min_data_in_leaf():
        return np.random.randint(min_data_in_leaf_range[0], min_data_in_leaf_range[1])

    def rand_learning_rate():
        return np.random.uniform(2, 3)

    def rand_num_leaf():
        return np.random.randint(num_leaf_range[0], num_leaf_range[1])

    def rand_lambda_l2():
        return np.random.uniform(1, 4)

    def write_to_file(line, label):
        f = open('temp_cv_res_random_month_%s.txt' % label, 'a')
        f.write(line + '\n')
        f.close()

    headers = ','.join(['%s-mean' % metric, '%s-stdv' % metric, 'n_rounds-mean', 'n_rounds_stdv', 'num_leaves', 'min_data_in_leaf', 'learning_rate'])
    for s in keep_size:
        write_to_file(headers, str(s))

    # gbms = []
    # for params_i in p.raw_lgb_2y:
    #     gbms.append(z.train_lgb(x_step1, y, z.get_params(params_i, 'reg')))
    #
    # error_step1 = y - z.pred_lgb_blend(x_step1, gbms)
    pred_step1_train = pkl.load(open('final_pred/pred_step1_train.pkl', 'rb'))
    error_step1 = y - pred_step1_train

    for i in range(1, n_iter + 1):
        rand_params = {'num_leaves': rand_num_leaf(),
                       'min_data_in_leaf': rand_min_data_in_leaf(),
                       'learning_rate': 0.1 ** rand_learning_rate(),
                       # 'lambda_l2': 0.1 ** rand_lambda_l2()
                       }
        params.update(rand_params)
        for s in keep_size:
            cv_hist = []
            n_rounds = []
            for mon_set in ({'01'}, {'02'}, {'03'}, {'04'}, {'05'}, {'06'}, {'07'}, {'08'}):
                for year in (2016, 2017):
                    use_idx = np.array(x_raw.index[np.logical_and(x_raw['sale_month'].apply(lambda x: x in mon_set), x_raw['data_year'] == year)])
                    if s == 'all':
                        pass
                    else:
                        np.random.shuffle(use_idx)
                        use_idx = use_idx[:s]
                    # print('train_size: %d' % int(np.sum(use_idx)))
                    pred_error = error_step1[use_idx]

                    train_x_step2_local = x_step2.loc[use_idx, :]
                    lgb_train = lgb.Dataset(train_x_step2_local, pred_error)
                    eval_hist = lgb.cv(params, lgb_train, stratified=False, num_boost_round=5000, early_stopping_rounds=100)
                    cv_hist.append([eval_hist['%s-mean' % metric][-1], eval_hist['%s-stdv' % metric][-1]])
                    n_rounds.append(len(eval_hist['%s-mean' % metric]))

            m_mean, m_stdv = np.array(cv_hist).mean(axis=0)
            n_rounds_mean = np.mean(np.array(n_rounds))
            n_rounds_stdv = np.std(np.array(n_rounds))
            line = '%.7f,%.7f,%.0f,%.0f,%.0f,%.0f,%.6f' % (m_mean,m_stdv, n_rounds_mean, n_rounds_stdv, rand_params['num_leaves'], rand_params['min_data_in_leaf'], rand_params['learning_rate'])
            write_to_file(line, str(s))
        print('finished %d / %d' % (i, n_iter))


def param_search_3step():
    error_series = pkl.load(open('error_after_month_train_2017.pkl', 'rb'))
    x = z.lgb_data_prep(z.train_2017_x, keep_only_feature=['2y_diff_dollar_taxvalue_total', '2y_diff_dollar_taxvalue_land', '2y_diff_dollar_taxvalue_structure'])

    n_iter = 50
    params_reg = z.params_base.copy()
    params_reg.update(z.params_reg)

    z.search_lgb_random(x, error_series, params_reg, 'lgb_step2_error_2017', n_iter)


def pred_month_2step():
    pred1 = z.pred_train_mon_2step()
    z.submit_nosea(pred1, 1)
    pred1 = z.pred_train_mon_2step(True)
    z.submit_nosea(pred1, 2)


def fe3():
    z.feature_engineering3_combined()


def train_prop():
    gbms_step1 = pkl.load(open('final_pred/gbms_step1.pkl', 'rb'))

    prop_2016_step1 = z.lgb_data_prep(z.prop_2016, p.class3_new_features, p.class3_rm_features)
    pred_2016_step1 = z.pred_lgb_blend(prop_2016_step1, gbms_step1)
    pkl.dump(pred_2016_step1, open('final_pred/pred_step1_2016.pkl', 'wb'))

    prop_2017_step1 = z.lgb_data_prep(z.prop_2017, p.class3_new_features, p.class3_rm_features)
    pred_2017_step1 = z.pred_lgb_blend(prop_2017_step1, gbms_step1)
    pkl.dump(pred_2017_step1, open('final_pred/pred_step1_2017.pkl', 'wb'))


if __name__ == '__main__':
    # param_search_batch()
    # fe3()
    # param_search_batch_one_mon()
    # pred_month_2step()
    # param_search_batch_with_outlier()
    # param_search_batch_one_mon((11000, 5500))
    # param_search_raw_lgb_final()
    # train_prop()
    # final_pred_script.pred_func()
    # only_catboost.sub_train()
    # only_catboost.main_fun()
    only_catboost.main_fun_v3()
