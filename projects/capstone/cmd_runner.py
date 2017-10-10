import pandas as pd
import numpy as np
import zillow_1 as z


def param_search_batch():
    prop_2016, prop_2017 = z.load_prop_data()
    train_x, train_y = z.load_train_data(prop_2016, prop_2017)
    keep_feature = list(z.feature_info.index.values)
    for col in ['census_block', 'raw_census_block']:
        keep_feature.remove(col)
    # for col in ['num_bathroom_assessor', 'code_county', 'area_living_finished_calc', 'area_firstfloor_assessor']:
    #     keep_feature.remove(col)
    train_x = train_x[keep_feature]
    train_x.index = list(range(train_x.shape[0]))

    z.prep_for_lgb_single(train_x)
    train_x_lgb = z.lgb_data_prep(train_x)

    n_iter = 50
    params_reg = z.params_base.copy()
    params_reg.update(z.params_reg)
    params_clf = z.params_base.copy()
    params_clf.update(z.params_clf)

    # raw lgb
    z.search_lgb_random(train_x_lgb, train_y, params_reg, 'raw_lgb', n_iter)

    # clf
    train_y_local = np.zeros(train_y.shape)
    train_y_local[train_y > 0] = 1
    z.search_lgb_random(train_x_lgb, train_y_local, params_clf, 'sign_error_clf', n_iter)

    # pos error
    mark_idx = train_y > 0
    train_x_local = train_x_lgb.loc[train_x_lgb.index[mark_idx], :]
    train_y_local = train_y.loc[mark_idx]
    z.search_lgb_random(train_x_local, train_y_local, params_reg, 'sign_error_pos', n_iter)

    # neg error
    mark_idx = train_y < 0
    train_x_local = train_x_lgb.loc[train_x_lgb.index[mark_idx], :]
    train_y_local = train_y.loc[mark_idx]
    z.search_lgb_random(train_x_local, train_y_local, params_reg, 'sign_error_neg', n_iter)

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


def fe3():
    z.feature_engineering3_combined()


if __name__ == '__main__':
    # param_search_batch()
    fe3()

