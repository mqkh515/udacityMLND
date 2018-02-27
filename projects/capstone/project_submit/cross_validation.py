from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def cv_avg(x, y, model_obj, n_folds=5):
    """testing set is selected randomly from indexes. If to be stratified by a column, the Series data should be provided.
       input model_func take a general API, takes in train_x and train_y and test_x, returns test_y， which columns to use should be handled inside model_func"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=7)
    evals = []
    for train_index, test_index in kf.split(x):
        train_x = x.loc[train_index, :].copy()
        train_y = y[train_index].copy()
        test_x = x.loc[test_index, :].copy()
        test_y = y.loc[test_index].copy()
        pred_y = model_obj.cv(train_x, train_y, test_x)
        evals.append(np.mean(np.abs(test_y.values - pred_y)))
    return np.mean(evals)


def cv_avg_stratified(x, y, model_obj, stratify_by='sale_month', n_folds=5):
    """stratified version of cv_avg, stratified by is a columns of x"""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
    evals = []
    for train_index, test_index in kf.split(x, x[stratify_by]):
        train_x = x.loc[train_index, :].copy()
        train_y = y[train_index].copy()
        test_x = x.loc[test_index, :].copy()
        test_y = y.loc[test_index].copy()
        pred_y = model_obj.cv(train_x, train_y, test_x)
        evals.append(np.mean(np.abs(test_y.values - pred_y)))
    return np.mean(evals)


def cv_public_lb(x, y, model_obj, months_set=(4, 5, 6), train_fraction=(0.5, 0.2, 0.2), n_folds=5):
    """testing set is created to mimic public LB behavior, i.e. given a months set out of 1 ~ 9, keep part of the 2016-months-set data in training, rest as testing."""
    evals = []
    for i in range(n_folds):
        train_index = x.index[~x['sale_month'].apply(lambda x: x in months_set)].tolist()  # 2017 data for target months also not included
        test_index = []
        np.random.seed(7 + i)
        for mon_i in range(len(months_set)):
            mon_index = x.index[np.logical_and(x['sale_year'] == 2016, x['sale_month'] == months_set[mon_i])].values
            mon_train_flag = np.random.rand(mon_index.shape[0]) < train_fraction[mon_i]
            train_index += mon_index[mon_train_flag].tolist()
            test_index += mon_index[~mon_train_flag].tolist()
        train_x = x.loc[train_index, :].copy()
        train_y = y[train_index].copy()
        test_x = x.loc[test_index, :].copy()
        test_y = y.loc[test_index].copy()
        pred_y = model_obj.cv(train_x, train_y, test_x)
        evals.append(np.mean(np.abs(test_y.values - pred_y)))
    return np.mean(evals)


def cv_private_lb(x, y, model_obj, months_set=(4, 5, 6), train_fraction=(0.5, 0.2, 0.2), n_folds=5):
    """testing set is created to mimic private LB behavior, i.e.  a quarter out of months 1 ~ 9, keep part of 2016-months-set data in training, discard the rest, and use all 2017-months-set data in testing"""
    evals = []
    for i in range(n_folds):
        train_index = x.index[~x['sale_month'].apply(lambda x: x in months_set)].tolist()
        test_index = []
        np.random.seed(7 + i)
        for mon_i in range(len(months_set)):
            # use part of 2016 for train, rest not used
            mon_index = x.index[np.logical_and(x['sale_year'] == 2016, x['sale_month'] == months_set[mon_i])].values
            mon_train_flag = np.random.rand(mon_index.shape[0]) < train_fraction[mon_i]
            train_index += mon_index[mon_train_flag].tolist()
            # use corresponding 2017 month for testing
            test_index += x.index[np.logical_and(x['sale_year'] == 2017, x['sale_month'] == months_set[mon_i])].values.tolist()
        train_x = x.loc[train_index, :].copy()
        train_y = y[train_index].copy()
        test_x = x.loc[test_index, :].copy()
        test_y = y.loc[test_index].copy()
        pred_y = model_obj.cv(train_x, train_y, test_x)
        evals.append(np.mean(np.abs(test_y.values - pred_y)))
    return np.mean(evals)
