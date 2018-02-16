from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import models
import cat_boost_models as cb_models
import lgb_models


def cv_avg(x, y, model_func, n_folds=5):
    """testing set is selected randomly from indexes. If to be stratified by a column, the Series data should be provided.
       input model_func take a general API, takes in train_x and train_y and test_x, returns test_y， which columns to use should be handled inside model_func"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=7)
    evals = []
    for train_index, test_index in kf.split(x):
        train_x = x.loc[train_index, :].copy()
        train_y = y[train_index].copy()
        test_x = x.loc[test_index, :].copy()
        test_y = y.loc[test_index].copy()
        pred_y = model_func(train_x, train_y, test_x)
        evals.append(np.mean(np.abs(test_y.values - pred_y)))
    return np.mean(evals)


def cv_avg_stratified(x, y, model_func, stratify_by='sale_month', n_folds=5):
    """stratified version of cv_avg, stratified by is a columns of x"""
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
    evals = []
    for train_index, test_index in kf.split(x, x[stratify_by]):
        train_x = x.loc[train_index, :].copy()
        train_y = y[train_index].copy()
        test_x = x.loc[test_index, :].copy()
        test_y = y.loc[test_index].copy()
        pred_y = model_func(train_x, train_y, test_x)
        evals.append(np.mean(np.abs(test_y.values - pred_y)))
    return np.mean(evals)


def cv_month(x, y, model_func, months_set={7, 8, 9}, n_folds=5):
    """testing set is created to mimic public LB behavior, i.e. given a months set out of 1 ~ 9, keep part of the 2016-months-set data in training, rest as testing."""
    np.random.seed(7)


def cv_year(x, y, model_func, months_set={7, 8, 9}, n_folds=5):
    """testing set is created to mimic private LB behavior, i.e.  a quarter out of months 1 ~ 9, keep part of 2016-months-set data in training, discard the rest, and use all 2017-months-set data in testing"""
    np.random.seed(7)

