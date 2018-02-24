import numpy as np
import data_prep
import pandas as pd
import cross_validation as cv
import lgb_models
import pickle as pkl
import os
import gc


def get_lb_rank(score, lb_type):
    df = pd.read_csv('data/%s_lb.csv' % lb_type)
    return df['rank'].values[np.searchsorted(df['score'].values, score)]


class ModelBase(object):
    def __init__(self):
        self.public_lb_score = None
        self.private_lb_score = None
        self.model = None

    def train(self, train_x, train_y, dump_model=True, load_model=False):
        pass

    def predict(self, test_x):
        pass

    def cv(self, train_x, train_y, test_x):
        model_backup = self.model
        self.train(train_x, train_y, False, False)
        pred = self.predict(test_x)
        self.model = model_backup
        return pred

    def analysis(self):
        # call this after self.public_lb_score and self.private_lb_score been manually filled
        cv_avg = cv.cv_avg_stratified(data_prep.train_x, data_prep.train_y, self)
        cv_public_lb = cv.cv_public_lb(data_prep.train_x, data_prep.train_y, self)
        cv_private_lb = cv.cv_private_lb(data_prep.train_x, data_prep.train_y, self)
        public_lb_rank = get_lb_rank(self.public_lb_score, 'public')
        private_lb_rank = get_lb_rank(self.private_lb_score, 'private')
        return [cv_avg, cv_private_lb, cv_public_lb, self.public_lb_score, public_lb_rank, self.private_lb_score, private_lb_rank]


class ModelMedian(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.public_lb_score = 0.0653607
        self.private_lb_score = 0.0763265

    def train(self, train_x, train_y, dump_model=True, load_model=False):
        if load_model and os.path.exists('models/model_median.pkl'):
            self.model = pkl.load(open('models/model_median.pkl', 'rb'))
        else:
            self.model = train_y.median()
            if dump_model:
                pkl.dump(self.model, open('models/model_median.pkl', 'wb'))

    def predict(self, test_x):
        return np.ones(test_x.shape[0]) * self.model

    def submit(self):
        submission = pd.read_csv('data/sample_submission.csv', header=0)
        df = pd.DataFrame()
        df['ParcelId'] = submission['ParcelId']
        if self.model is None:
            self.train(data_prep.train_x, data_prep.train_y)
        median_pred = self.predict(data_prep.prop_2016)
        df['201610'] = median_pred
        df['201611'] = median_pred
        df['201612'] = median_pred
        df['201710'] = median_pred
        df['201711'] = median_pred
        df['201712'] = median_pred
        df.to_csv('data/submission_model_median.csv.gz', index=False, float_format='%.7f', compression='gzip')


class ModelLGBRaw(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.params = {'num_leaves': 76,
                       'min_data_in_leaf': 168,
                       'learning_rate': 0.0062,
                       'num_boosting_rounds': 2250
                       }
        self.public_lb_score = 0.0641573
        self.private_lb_score = 0.0750084
        self.added_features = []  # for FE

    def train(self, train_x, train_y, dump_model=True, load_model=False):
        if load_model and os.path.exists('models/model_median.pkl'):
            self.model = pkl.load(open('models/model_median.pkl', 'rb'))
        else:
            train_x['sale_month'] = train_x['sale_month'].apply(lambda x: x if x < 10 else 10)
            train_x = lgb_models.lgb_model_prep(train_x, self.added_features + ['sale_month'])
            params = lgb_models.PARAMS.copy()
            params.update(self.params)
            self.model = lgb_models.train_lgb(train_x, train_y, params)
            if dump_model:
                pkl.dump(self.model, open('models/model_lgb_raw.pkl', 'wb'))

    def predict(self, test_x):
        test_x = lgb_models.lgb_model_prep(test_x, self.added_features + ['sale_month'])
        return self.model.predict(test_x)

    def submit(self):
        submission = pd.read_csv('data/sample_submission.csv', header=0)
        df = pd.DataFrame()
        df['ParcelId'] = submission['ParcelId']
        if self.model is None:
            self.train(data_prep.train_x, data_prep.train_y)
        prop_2016 = data_prep.prop_2016.copy()
        prop_2016['sale_month'] = 10
        pred_2016 = self.predict(prop_2016)
        df['201610'] = pred_2016
        df['201611'] = pred_2016
        df['201612'] = pred_2016
        del prop_2016
        gc.collect()

        prop_2017 = data_prep.prop_2017.copy()
        prop_2017['sale_month'] = 10
        pred_2017 = self.predict(prop_2017)
        df['201710'] = pred_2017
        df['201711'] = pred_2017
        df['201712'] = pred_2017
        df.to_csv('data/submission_model_lgb_raw.csv.gz', index=False, float_format='%.7f', compression='gzip')

