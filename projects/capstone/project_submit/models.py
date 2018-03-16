import numpy as np
import data_prep
import pandas as pd
import cross_validation as cv
import lgb_models
import pickle as pkl
import os
import gc
import params
import cat_boost_models


def format_cv_output(m):
    out_data = m.analysis()
    pd.options.display.float_format = '{:,.7f}'.format
    df = pd.DataFrame([out_data], columns=['cv_avg',
                                           'cv_public_LB',
                                           'cv_private_LB',
                                           'score_public_LB',
                                           'rank_public_LB',
                                           'score_private_LB',
                                           'rank_private_LB'])
    df.to_csv('models/cv_res_%s.csv' % m.label)


def get_lb_rank(score, lb_type):
    df = pd.read_csv('data/%s_lb.csv' % lb_type)
    return df['rank'].values[np.searchsorted(df['score'].values, score)]


def submit_sep_month(model):
    submission = pd.read_csv('data/sample_submission.csv', header=0)
    df = pd.DataFrame()
    df['ParcelId'] = submission['ParcelId']
    model.train(data_prep.train_x, data_prep.train_y)
    prop_2016 = data_prep.prop_2016.copy()
    for m in [10, 11, 12]:
        prop_2016['sale_month'] = m
        pred_2016 = model.predict(prop_2016)
        k = '2016' + str(m)
        df[k] = pred_2016
    del prop_2016
    gc.collect()
    prop_2017 = data_prep.prop_2017.copy()
    for m in [10, 11, 12]:
        prop_2017['sale_month'] = m
        pred_2017 = model.predict(prop_2017)
        k = '2017' + str(m)
        df[k] = pred_2017
    df.to_csv('data/submission_model_%s.csv.gz' % model.label, index=False, float_format='%.7f', compression='gzip')


def submit_combine_month(model):
    """combine month for raw lgb, let's make this faster"""
    submission = pd.read_csv('data/sample_submission.csv', header=0)
    df = pd.DataFrame()
    df['ParcelId'] = submission['ParcelId']
    model.train(data_prep.train_x, data_prep.train_y)
    prop_2016 = data_prep.prop_2016.copy()
    prop_2016['sale_month'] = 10
    pred_2016 = model.predict(prop_2016)
    df['201610'] = pred_2016
    df['201611'] = pred_2016
    df['201612'] = pred_2016
    del prop_2016
    gc.collect()
    prop_2017 = data_prep.prop_2017.copy()
    prop_2017['sale_month'] = 10
    pred_2017 = model.predict(prop_2017)
    df['201710'] = pred_2017
    df['201711'] = pred_2017
    df['201712'] = pred_2017
    df.to_csv('data/submission_model_%s.csv.gz' % model.label, index=False, float_format='%.7f', compression='gzip')


class ModelBase(object):
    def __init__(self):
        self.public_lb_score = 0.0
        self.private_lb_score = 0.0
        self.model = None
        self.label = ''
        self.outlier_handling = None

    def train(self, train_x, train_y, dump_model=True, load_model=False):
        if load_model and os.path.exists('models/model_%s.pkl' % self.label):
            self.model = pkl.load(open('models/model_%s.pkl' % self.label, 'rb'))
        else:
            if self.outlier_handling:
                train_x, train_y = self.outlier_handling(train_x, train_y)  # key line, remove outliers before training
            self.train_inner(train_x, train_y)
            if dump_model:
                pkl.dump(self.model, open('models/model_%s.pkl' % self.label, 'wb'))

    def train_inner(self, train_x, train_y):
        """set self.model in this function call"""
        pass

    def predict(self, test_x):
        pass

    def cv_prep(self, train_x):
        pass

    def cv(self, train_x, train_y, test_x):
        model_backup = self.model
        self.cv_prep(train_x)
        self.train(train_x, train_y, False, False)
        self.cv_prep(test_x)
        pred = self.predict(test_x)
        self.model = model_backup
        return pred

    def analysis(self):
        # call this after self.public_lb_score and self.private_lb_score been manually filled
        cv_avg = cv.cv_avg_stratified(data_prep.train_x, data_prep.train_y, self)
        cv_public_lb = cv.cv_public_lb(data_prep.train_x, data_prep.train_y, self)
        cv_private_lb = cv.cv_private_lb(data_prep.train_x, data_prep.train_y, self)
        public_lb_rank = get_lb_rank(self.public_lb_score, 'public') if self.public_lb_score else 0
        private_lb_rank = get_lb_rank(self.private_lb_score, 'private') if self.private_lb_score else 0
        return [cv_avg, cv_private_lb, cv_public_lb, self.public_lb_score, public_lb_rank, self.private_lb_score, private_lb_rank]

    def submit(self):
        submit_sep_month(self)


class ModelMedian(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.public_lb_score = 0.0653607
        self.private_lb_score = 0.0763265
        self.label = 'median'

    def train_inner(self, train_x, train_y):
        self.model = train_y.median()

    def predict(self, test_x):
        return np.ones(test_x.shape[0]) * self.model

    def submit(self):
        """much simpler implementation"""
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
        df.to_csv('data/submission_model_%s.csv.gz' % self.label, index=False, float_format='%.7f', compression='gzip')


class ModelLGBRaw(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.params = params.lgb_raw
        self.public_lb_score = 0.0643716
        self.private_lb_score = 0.0752874
        self.added_features = []
        self.rm_features = []
        self.keep_only_features = []
        self.label = 'lgb_raw'

    def prep_added_features(self, train_x):
        pass

    def train_inner(self, train_x, train_y):
        self.prep_added_features(train_x)
        train_x = lgb_models.lgb_model_prep(train_x, self.added_features, self.rm_features, self.keep_only_features)
        params = lgb_models.PARAMS.copy()
        params.update(self.params)
        self.model = lgb_models.train_lgb(train_x, train_y, params)

    def predict(self, test_x):
        self.prep_added_features(test_x)
        test_x = lgb_models.lgb_model_prep(test_x, self.added_features, self.rm_features, self.keep_only_features)
        return self.model.predict(test_x)

    def submit(self):
        submit_combine_month(self)


class ModelLGBRawSubCol(ModelLGBRaw):
    def __init__(self):
        ModelLGBRaw.__init__(self)
        self.public_lb_score = 0.0644800
        self.private_lb_score = 0.0753710
        self.rm_features = []
        for col in list(lgb_models.feature_info.index.values):
            if lgb_models.feature_info.loc[col, 'class'] in (3, 4):
                col_name = col if lgb_models.feature_info.loc[col, 'type'] == 'num' else col + '_lgb'
                self.rm_features.append(col_name)
        self.label = 'lgb_raw_sub_col'


class ModelLGBRawIncMon(ModelLGBRaw):
    def __init__(self):
        """combined month version of RawLGB"""
        ModelLGBRaw.__init__(self)
        self.public_lb_score = 0.0641573
        self.private_lb_score = 0.0750084
        self.added_features = ['sale_month_derive']
        self.label = 'lgb_raw_inc_mon'
        self.target_mon = set()
        self.target_mon_label = None

    def prep_added_features(self, train_x):
        train_x['sale_month_derive'] = train_x['sale_month'].apply(lambda x: x if x not in self.target_mon else self.target_mon_label)

    def cv_prep(self, train_x):
        self.target_mon = {4, 5, 6}
        self.target_mon_label = 4

    def submit(self):
        self.target_mon = {10, 11, 12}
        self.target_mon_label = 10
        submit_combine_month(self)


class ModelLGBRawIncMonOutlierRm(ModelLGBRawIncMon):
    def __init__(self):
        ModelLGBRawIncMon.__init__(self)
        self.public_lb_score = 0.0641336
        self.private_lb_score = 0.0749581
        self.label = 'lgb_raw_inc_mon_outlier_rm'
        self.outlier_handling = data_prep.rm_outlier


class ModelLGBOneStep(ModelLGBRawIncMonOutlierRm):
    def __init__(self):
        ModelLGBRawIncMonOutlierRm.__init__(self)
        self.public_lb_score = 0.0641632
        self.private_lb_score = 0.0749579
        self.added_features += params.class3_new_features
        self.rm_features = params.class3_rm_features
        self.label = 'lgb_1step'


class ModelLGBOneStepDefaultParam(ModelLGBOneStep):
    def __init__(self):
        ModelLGBOneStep.__init__(self)
        self.public_lb_score = 0.0643160
        self.private_lb_score = 0.0752334
        self.added_features += params.class3_new_features
        self.rm_features = params.class3_rm_features
        self.label = 'lgb_1step_default_param'
        self.params = params.lgb_default


# class ModelLGBOneStepSepMonth(ModelLGBOneStep):
#     def __init__(self):
#         ModelLGBOneStep.__init__(self)
#         self.public_lb_score = 0.0641749
#         self.private_lb_score = 0.0749703
#         self.added_features = ['sale_month'] + params.class3_new_features
#         self.rm_features = params.class3_rm_features
#         self.label = 'lgb_1step_sep_month'
#
#     def prep_added_features(self, train_x):
#         pass
#
#     def cv_prep(self, train_x):
#         pass
#
#     def submit(self):
#         submit_sep_month(self)


class ModelLGBBlending(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.label = ''  # not expected to be used as a real model
        self.model = None
        self.model_setup()
        self.outlier_handling = data_prep.rm_outlier

    def model_setup(self):
        pass

    def train_inner(self, train_x, train_y):
        for m in self.model:
            m.train(train_x, train_y, False, False)

    def predict(self, test_x):
        pred = None
        for m in self.model:
            pred_m = m.predict(test_x)
            if pred is None:
                pred = pred_m
            else:
                pred += pred_m
        return pred / len(self.model)

    def submit(self):
        submit_combine_month(self)


class ModelLGBBlendingOneStep(ModelLGBBlending):
    def __init__(self):
        ModelLGBBlending.__init__(self)
        self.public_lb_score = 0.0641527
        self.private_lb_score = 0.0749282
        self.label = 'lgb_1step_blending'

    def model_setup(self):
        self.model = []
        for p in params.lgb_step1:
            m = ModelLGBOneStep()
            m.params = p
            self.model.append(m)

    def submit(self):
        for m in self.model:
            m.target_mon = {10, 11, 12}
            m.target_mon_label = 10
        submit_combine_month(self)


# class ModelLGBBlendingOneStepSepMonth(ModelLGBBlending):
#     def __init__(self):
#         ModelLGBBlending.__init__(self)
#         self.public_lb_score = 0.0641476
#         self.private_lb_score = 0.0749362
#         self.label = 'lgb_1step_sep_month_blending'
#
#     def model_setup(self):
#         self.model = []
#         for p in params.lgb_step1:
#             m = ModelLGBOneStepSepMonth()
#             m.params = p
#             self.model.append(m)
#
#     def submit(self):
#         submit_sep_month(self)


# class ModelLGBBlendingOneStepMoreRounds(ModelLGBBlendingOneStep):
#     def __init__(self):
#         ModelLGBBlendingOneStep.__init__(self)
#         self.public_lb_score = 0.0
#         self.private_lb_score = 0.0
#         self.label = 'lgb_1step_blending_more_rounds'
#         for m in self.model:
#             m.params['num_boosting_rounds'] += int(m.params['num_boosting_rounds'] * 0.1)


class ModelLGBBlendingTwoStepStep1(ModelLGBBlending):
    def __init__(self):
        ModelLGBBlending.__init__(self)
        self.label = ''

    def model_setup(self):
        self.model = []
        for p in params.lgb_step1:
            m = ModelLGBRaw()
            m.params = p
            m.added_features = params.class3_new_features
            m.rm_features = params.class3_rm_features
            m.outlier_handling = data_prep.rm_outlier
            self.model.append(m)


class ModelLGBBlendingTwoStepStep2(ModelLGBBlending):
    def __init__(self):
        ModelLGBBlending.__init__(self)
        self.label = ''

    def model_setup(self):
        self.model = []
        for p in params.lgb_step2:
            m = ModelLGBRaw()
            m.params = p
            m.keep_only_features = params.step2_keep_only_feature
            m.outlier_handling = data_prep.rm_outlier
            self.model.append(m)


class ModelLGBTwoStepBase(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.public_lb_score = 0.0
        self.private_lb_score = 0.0
        self.label = ''
        self.model = None
        self.model_setup()
        self.outlier_handling = data_prep.rm_outlier
        self.target_mon = set()  # save the spot, target_mon could be different in cv and final predict

    def model_setup(self):
        pass

    def cv_prep(self, train_x):
        self.target_mon = {4, 5, 6}

    def train_inner(self, train_x, train_y):
        assert self.target_mon
        self.model['step1'].train(train_x, train_y, False, False)
        pred1 = self.model['step1'].predict(train_x)
        error1 = train_y - pred1
        # CV should make sure only 2016 data is used in training
        if np.any(np.logical_and(train_x['sale_month'].apply(lambda x: x in self.target_mon), train_x['data_year'] == 2017)):
            raise Exception('2017 target months data used in training')
        tar_index = np.array(train_x.index[train_x['sale_month'].apply(lambda x: x in self.target_mon)])
        assert np.all(train_x.index == error1.index)
        train_x_step2 = train_x.loc[tar_index, :]
        train_y_step2 = error1[tar_index]
        self.model['step2'].train(train_x_step2, train_y_step2, False, False)

    def predict(self, test_x):
        pred_step1 = self.model['step1'].predict(test_x)
        pred_step2 = self.model['step2'].predict(test_x)
        return pred_step1 + pred_step2

    def analysis(self):
        # no avg CV for 2-step lgb
        cv_public_lb = cv.cv_public_lb(data_prep.train_x, data_prep.train_y, self)
        cv_private_lb = cv.cv_private_lb(data_prep.train_x, data_prep.train_y, self)
        public_lb_rank = get_lb_rank(self.public_lb_score, 'public') if self.public_lb_score else 0
        private_lb_rank = get_lb_rank(self.private_lb_score, 'private') if self.private_lb_score else 0
        return [0.0, cv_private_lb, cv_public_lb, self.public_lb_score, public_lb_rank, self.private_lb_score, private_lb_rank]

    def submit(self):
        self.target_mon = {10, 11, 12}
        submit_combine_month(self)


class ModelLGBTwoStep(ModelLGBTwoStepBase):
    def __init__(self):
        ModelLGBTwoStepBase.__init__(self)
        self.public_lb_score = 0.0641599
        self.private_lb_score = 0.0749253
        self.label = 'lgb_2step'

    def model_setup(self):
        m_step1 = ModelLGBRaw()
        m_step1.params = params.lgb_step1_1
        m_step1.added_features = params.class3_new_features
        m_step1.rm_features = params.class3_rm_features
        m_step2 = ModelLGBRaw()
        m_step2.params = params.lgb_step2_1
        m_step2.keep_only_features = params.step2_keep_only_feature
        self.model = {'step1': m_step1,
                      'step2': m_step2}


class ModelLGBTwoStepBlending(ModelLGBTwoStepBase):
    def __init__(self):
        ModelLGBTwoStepBase.__init__(self)
        self.public_lb_score = 0.0641384
        self.private_lb_score = 0.0749144
        self.label = 'lgb_2step_blending'

    def model_setup(self):
        self.model = {'step1': ModelLGBBlendingTwoStepStep1(),
                      'step2': ModelLGBBlendingTwoStepStep2()}


# class ModelLGBTwoStepBlendingMoreRounds(ModelLGBTwoStepBlending):
#     def __init__(self):
#         ModelLGBTwoStepBlending.__init__(self)
#         for m in self.model['step1'].model:
#             m.params['num_boosting_rounds'] += int(m.params['num_boosting_rounds'] * 0.1)
#         self.public_lb_score = 0.0
#         self.private_lb_score = 0.0
#         self.label = 'lgb_2step_blending_more_rounds'


class ModelCatBoost(ModelBase):
    def __init__(self):
        ModelBase.__init__(self)
        self.public_lb_score = 0.0641350
        self.private_lb_score = 0.0748987
        self.label = 'catboost'
        self.outlier_handling = data_prep.rm_outlier
        self.target_mon = set()
        self.target_mon_label = None
        self.seed = 1
        self.params = params.catboost_tuned

    def cv_prep(self, train_x):
        self.target_mon = {4, 5, 6}
        self.target_mon_label = 4

    def train_inner(self, train_x, train_y):
        train_x['sale_month_derive'] = train_x['sale_month'].apply(lambda x: x if x not in self.target_mon else self.target_mon_label)
        train_x, cat_inds = cat_boost_models.cat_boost_model_prep(train_x, ['sale_month_derive'], [False])
        m = cat_boost_models.cat_boost_obj_gen(self.params, self.seed)
        m.fit(train_x, train_y, cat_features=cat_inds)
        self.model = m

    def predict(self, test_x):
        test_x['sale_month_derive'] = test_x['sale_month'].apply(lambda x: x if x not in self.target_mon else self.target_mon_label)
        test_x, _ = cat_boost_models.cat_boost_model_prep(test_x, ['sale_month_derive'], [False])
        pred = self.model.predict(test_x)
        return pred

    def submit(self):
        self.target_mon = {10, 11, 12}
        self.target_mon_label = 10
        submit_combine_month(self)


class ModelCatBoostDefaultParam(ModelCatBoost):
    def __init__(self):
        ModelCatBoost.__init__(self)
        self.public_lb_score = 0.0642229
        self.private_lb_score = 0.0750398
        self.label = 'catboost_default_param'
        self.outlier_handling = data_prep.rm_outlier
        self.params = params.catboost_default


class ModelCatBoostBlending(ModelLGBBlending):
    def __init__(self):
        ModelLGBBlending.__init__(self)
        self.public_lb_score = 0.0640620
        self.private_lb_score = 0.0748351
        self.label = 'cat_boost_blending'

    def model_setup(self):
        self.model = []
        for s in range(1, 6):
            m = ModelCatBoost()
            m.seed = s
            self.model.append(m)

    def analysis(self):
        # no CV
        public_lb_rank = get_lb_rank(self.public_lb_score, 'public') if self.public_lb_score else 0
        private_lb_rank = get_lb_rank(self.private_lb_score, 'private') if self.private_lb_score else 0
        return [0.0, 0.0, 0.0, self.public_lb_score, public_lb_rank, self.private_lb_score, private_lb_rank]

    def submit(self):
        for m in self.model:
            m.target_mon = {10, 11, 12}
            m.target_mon_label = 10
        submit_combine_month(self)


class ModelDirectBlending:
    def __init__(self):
        self.public_lb_score = 0.0
        self.private_lb_score = 0.0
        self.label = ''
        self.elements = []
        self.elements_weight = []

    def analysis(self):
        # no CV
        public_lb_rank = get_lb_rank(self.public_lb_score, 'public') if self.public_lb_score else 0
        private_lb_rank = get_lb_rank(self.private_lb_score, 'private') if self.private_lb_score else 0
        return [0.0, 0.0, 0.0, self.public_lb_score, public_lb_rank, self.private_lb_score, private_lb_rank]

    def submit(self):
        pred = None
        columns = ['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']
        for idx, f in enumerate(self.elements):
            p = pd.read_csv('data/submission_model_%s.csv.gz' % f, header=0, compression='gzip')
            p = p.loc[:, columns]
            if pred is None:
                pred = p * self.elements_weight[idx]
            else:
                pred += p * self.elements_weight[idx]
        pred /= np.sum(self.elements_weight)
        pred['ParcelId'] = pred['ParcelId'].astype(int)
        pred.to_csv('data/submission_model_%s.csv.gz' % self.label, index=False, float_format='%.7f', compression='gzip')


class ModelDirectBlendingLGB(ModelDirectBlending):
    def __init__(self):
        ModelDirectBlending.__init__(self)
        self.public_lb_score = 0.0641301
        self.private_lb_score = 0.0748995
        self.label = 'blending_lgb'
        self.elements = ['lgb_1step_blending', 'lgb_2step_blending']
        self.elements_weight = [1.0, 1.0]


class ModelDirectBlendingAll(ModelDirectBlending):
    def __init__(self):
        ModelDirectBlending.__init__(self)
        self.public_lb_score = 0.0640158
        self.private_lb_score = 0.0747848
        self.label = 'blending_all'
        self.elements = ['lgb_1step_blending', 'lgb_2step_blending', 'cat_boost_blending']
        self.elements_weight = [1.0, 1.0, 2.0]









