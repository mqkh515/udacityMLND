import zillow_1 as z
import lightgbm as lgb

params_clf = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 48,
    'min_data_in_leaf': 200,
    'learning_rate': 0.0045,
    'lambda_l2': 0.004,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbosity': 0,
    'num_boosting_rounds': 2200
}


def error_clf_data_prep(train_data, test_data):
    pass


def error_clf_train(train_x, train_y):
    pass


def error_clf_cv(train_x, train_y):
    pass


params_small_error = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': {'l1'},
    'num_leaves': 48,
    'min_data_in_leaf': 200,
    'learning_rate': 0.0045,
    'lambda_l2': 0.004,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbosity': 0,
    'num_boosting_rounds': 2200
}


def small_error_data_perp(train_data, test_data):
    pass


params_big_error = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l1',
    'metric': {'l1'},
    'num_leaves': 48,
    'min_data_in_leaf': 200,
    'learning_rate': 0.0045,
    'lambda_l2': 0.004,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbosity': 0,
    'num_boosting_rounds': 2200
}


def big_error_data_prep(train_data, test_data):
    pass


def model_2layer(train_x, train_y, test_x):
    pass

