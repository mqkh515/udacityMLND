import sklearn
import numpy as np
import pandas as pd
import models
import cat_boost_models as cb_models
import lgb_models


def cv_avg(x, y, model_func, stratify_by=None, n_folds=5):
    """testing set is selected randomly from indexes. If to be stratified by a column, the vector data should be provided.
       input model_func take a general API, takes in x and y, and returns predict y"""
    pass


def cv_month(x, y, model_func, months_set={7, 8, 9}, n_folds=5):
    """testing set is created to mimic public LB behavior, i.e. given a months set out of 1 ~ 9, keep part of the 2016-months-set data in training, rest as testing."""
    pass


def cv_year(x, y, model_func, months_set={7, 8, 9}, n_folds=5):
    """testing set is created to mimic private LB behavior, i.e.  a quarter out of months 1 ~ 9, keep part of 2016-months-set data in training, discard the rest, and use all 2017-months-set data in testing"""
    pass

