import zillow_1 as z
import params_cache as p
import pandas as pd
import numpy as np

outlier_handle = z.capfloor_outlier

step1_new_features = ()
step1_rm_features = ()

step2_new_features = ()
step2_rm_features = ()

step3_new_features = ()
step3_rm_features = ()

mon_sets = ({'01'}, {'02'}, {'03'}, {'04'}, {'05'}, {'06'}, {'07'}, {'08'}, {'09'}, {'10', '11', '12'})

new_features = set()
new_features = new_features.union(set(step1_new_features))
new_features = new_features.union(set(step2_new_features))
new_features = new_features.union(set(step3_new_features))

train_x, train_y = z.load_train_data(z.prop_2016, z.prop_2017, new_features)


def pred_step2(x_raw, error1, mon_set, params):
    idx = x_raw['sale_month'].apply(lambda x: x in mon_set)
    y_step2 = error1[idx.values]
    x_raw_step2 = x_raw.loc[x_raw.index[idx], :]
    x_step2 = z.lgb_data_prep(x_raw_step2, step2_new_features, step2_rm_features)
    gbms_step2 = []
    for param in params:
        gbms_step2.append(z.train_lgb(x_step2, y_step2, z.get_params(param, 'reg')))

    # predict for 2016
    prop_2016_step2 = z.lgb_data_prep(z.prop_2016, step2_new_features, step2_rm_features)
    pred_2016_step2 = z.pred_lgb_blend(prop_2016_step2, gbms_step2)

    # predict for 2017
    prop_2017_step2 = z.lgb_data_prep(z.prop_2017, step2_new_features, step2_rm_features)
    pred_2017_step2 = z.pred_lgb_blend(prop_2017_step2, gbms_step2)
    return pred_2016_step2, pred_2017_step2


def pred_func(sep_month_pred, do_year_pred):
    """2 step predict: raw + month"""
    # first raw prediction
    x_raw, y = outlier_handle(train_x, train_y)

    # raw prediction
    x = z.lgb_data_prep(x_raw, step1_new_features, step1_rm_features)
    gbms_step1 = []
    for param in p.raw_lgb_2y:
        gbms_step1.append(z.train_lgb(x, y, z.get_params(param, 'reg')))
    pred1_train = z.pred_lgb_blend(x, gbms_step1)
    error1 = y - pred1_train

    # year prediction
    if do_year_pred:
        # first collect all step2 errors for train month
        for mon_set in ({'01'}, {'02'}, {'03'}, {'04'}, {'05'}, {'06'}, {'07'}, {'08'}, {'09'}):



    if sep_month_pred:
        pass
    else:
        pred_2016_step2, pred_2017_step2 = pred_step2(x_raw, error1, {'10', '11', '12'}, p.lgb_month)

        # predict for 2016
        prop_2016_step1 = z.lgb_data_prep(z.prop_2016, step1_new_features, step1_rm_features)
        pred_2016_step1 = z.pred_lgb_blend(prop_2016_step1, gbms_step1)
        pred_2016 = pred_2016_step1 + pred_2016_step2

        # predict for 2017
        prop_2017_step1 = z.lgb_data_prep(z.prop_2017, step1_new_features, step1_rm_features)
        pred_2017_step1 = z.pred_lgb_blend(prop_2017_step1, gbms_step1)
        pred_2017 = pred_2017_step1 + pred_2017_step2
