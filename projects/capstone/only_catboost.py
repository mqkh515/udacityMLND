import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime as dt
import pickle as pkl


def main_fun():
    print('Loading Properties ...')
    properties2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    properties2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)

    print('Loading Train ...')
    train2016 = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
    train2017 = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

    def add_date_features(df):
        df["transaction_year"] = df["transactiondate"].dt.year
        df["transaction_month"] = df["transactiondate"].dt.month
        df["transaction_year_month"] = df["transaction_year"] * 100 + df["transaction_month"]
        df["transaction_quarter"] = df["transactiondate"].dt.quarter
        df["transaction_year_quarter"] = df["transaction_year"] * 10 + df["transaction_quarter"]
        df.drop(["transactiondate"], inplace=True, axis=1)
        return df

    def proc_census_block(prop_data):
        prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
        prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('temp', axis=1, inplace=True)
        prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

        # census_block
        prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
        prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('censustractandblock', axis=1, inplace=True)
        return prop_data

    train2016 = add_date_features(train2016)
    train2017 = add_date_features(train2017)

    properties2016 = proc_census_block(properties2016)
    properties2017 = proc_census_block(properties2017)

    print('Loading Sample ...')
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)

    print('Merge Train with Properties ...')
    train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
    train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')

    # print('Tax Features 2017  ...')
    # train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

    print('Concat Train 2016 & 2017 ...')
    train_df = pd.concat([train2016, train2017], axis=0)
    test_df_2016 = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')
    test_df_2017 = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')

    del properties2016, properties2017, train2016, train2017
    gc.collect()

    print('Remove missing data fields ...')

    missing_perc_thresh = 0.98
    exclude_missing = []
    num_rows = train_df.shape[0]
    for c in train_df.columns:
        num_missing = train_df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing.append(c)
    print("We exclude: %s" % len(exclude_missing))

    del num_rows, missing_perc_thresh
    gc.collect()

    print("Remove features with one unique value !!")
    exclude_unique = []
    for c in train_df.columns:
        num_uniques = len(train_df[c].unique())
        if train_df[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            exclude_unique.append(c)
    print("We exclude: %s" % len(exclude_unique))

    print("Define training features !!")
    exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
    train_features = []
    for c in train_df.columns:
        if c not in exclude_missing \
                and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % len(train_features))

    print("Define categorial features !!")
    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        if c in {'census', 'block', 'raw_census', 'raw_block'}:
            cat_feature_inds.append(i)
        else:
            num_uniques = len(train_df[c].unique())
            if num_uniques < cat_unique_thresh \
                    and not 'sqft' in c \
                    and not 'cnt' in c \
                    and not 'nbr' in c \
                    and not 'number' in c:
                cat_feature_inds.append(i)


    print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
    print("all train features are: %s" % ','.join(train_features))

    print("Replacing NaN values by -999 !!")
    train_df.fillna(-999, inplace=True)
    test_df_2016.fillna(-999, inplace=True)
    test_df_2017.fillna(-999, inplace=True)

    print("Training time !!")
    X_train = train_df[train_features]
    # pkl.dump(X_train, open('only_catboost_x_train_v2.pkl', 'wb'))
    y_train = train_df.logerror
    # pkl.dump(y_train, open('only_catboost_y_train_v2.pkl', 'wb'))
    print(X_train.shape, y_train.shape)

    num_ensembles = 2
    models = []
    for i in tqdm(range(num_ensembles)):
        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=i)
        model.fit(
            X_train, y_train,
            cat_features=cat_feature_inds)
        models.append(model)

    def pred_test_data(year, month):
        print('predicting for %s-%s' % (year, month))
        test_data_orig = test_df_2016 if year == '2016' else test_df_2017
        test_df = test_data_orig.copy()
        test_df['transactiondate'] = pd.Timestamp('%s-%s-01' % (year, month))
        test_df = add_date_features(test_df)
        X_test = test_df[train_features]
        y_pred = 0.0
        for m in models:
            y_pred += m.predict(X_test)
        y_pred /= num_ensembles
        del test_df
        gc.collect()
        print('prediction finished for %s-%s' % (year, month))
        return y_pred

    y_pred_2016_10 = pred_test_data('2016', '10')
    y_pred_2016_11 = pred_test_data('2016', '11')
    y_pred_2016_12 = pred_test_data('2016', '12')
    y_pred_2017_10 = pred_test_data('2017', '10')
    y_pred_2017_11 = pred_test_data('2017', '11')
    y_pred_2017_12 = pred_test_data('2017', '12')

    submission = pd.DataFrame({
        'ParcelId': test_df_2016['ParcelId'],
    })
    submission['201610'] = y_pred_2016_10
    submission['201611'] = y_pred_2016_11
    submission['201612'] = y_pred_2016_12
    submission['201710'] = y_pred_2017_10
    submission['201711'] = y_pred_2017_11
    submission['201712'] = y_pred_2017_12

    submission.to_csv('data/only_catboost_v3.csv.gz', float_format='%.7f', index=False, compression='gzip')


def main_fun_v2():
    print('Loading Properties ...')
    properties2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    properties2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)

    print('Loading Train ...')
    train2016 = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
    train2017 = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

    def add_date_features(df):
        df["transaction_year"] = df["transactiondate"].dt.year
        df["transaction_month"] = df["transactiondate"].dt.month
        df["transaction_month"] = df["transaction_month"].apply(lambda x: x if x <= 10 else 10)
        df["transaction_year_month"] = df["transaction_year"] * 100 + df["transaction_month"]
        df["transaction_quarter"] = df["transactiondate"].dt.quarter
        df["transaction_year_quarter"] = df["transaction_year"] * 10 + df["transaction_quarter"]
        df.drop(["transactiondate"], inplace=True, axis=1)
        return df

    def proc_census_block(prop_data):
        prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
        prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('temp', axis=1, inplace=True)
        prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

        # census_block
        prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
        prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('censustractandblock', axis=1, inplace=True)
        return prop_data

    train2016 = add_date_features(train2016)
    train2017 = add_date_features(train2017)

    properties2016 = proc_census_block(properties2016)
    properties2017 = proc_census_block(properties2017)

    print('Loading Sample ...')
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)

    print('Merge Train with Properties ...')
    train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
    train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')

    # print('Tax Features 2017  ...')
    # train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

    print('Concat Train 2016 & 2017 ...')
    train_df = pd.concat([train2016, train2017], axis=0)
    test_df_2016 = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')
    test_df_2017 = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')

    del properties2016, properties2017, train2016, train2017
    gc.collect()

    print('Remove missing data fields ...')

    missing_perc_thresh = 0.98
    exclude_missing = []
    num_rows = train_df.shape[0]
    for c in train_df.columns:
        num_missing = train_df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing.append(c)
    print("We exclude: %s" % len(exclude_missing))

    del num_rows, missing_perc_thresh
    gc.collect()

    print("Remove features with one unique value !!")
    exclude_unique = []
    for c in train_df.columns:
        num_uniques = len(train_df[c].unique())
        if train_df[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            exclude_unique.append(c)
    print("We exclude: %s" % len(exclude_unique))

    print("Define training features !!")
    exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
    train_features = []
    for c in train_df.columns:
        if c not in exclude_missing \
                and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % len(train_features))

    print("Define categorial features !!")
    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        if c in {'census', 'block', 'raw_census', 'raw_block'}:
            cat_feature_inds.append(i)
        else:
            num_uniques = len(train_df[c].unique())
            if num_uniques < cat_unique_thresh \
                    and not 'sqft' in c \
                    and not 'cnt' in c \
                    and not 'nbr' in c \
                    and not 'number' in c:
                cat_feature_inds.append(i)

    print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
    print("all train features are: %s" % ','.join(train_features))

    print("Replacing NaN values by -999 !!")
    train_df.fillna(-999, inplace=True)
    test_df_2016.fillna(-999, inplace=True)
    test_df_2017.fillna(-999, inplace=True)

    print("Training time !!")
    X_train = train_df[train_features]
    # pkl.dump(X_train, open('only_catboost_x_train_v2.pkl', 'wb'))
    y_train = train_df.logerror
    # pkl.dump(y_train, open('only_catboost_y_train_v2.pkl', 'wb'))
    print(X_train.shape, y_train.shape)

    test_df_2016_10 = test_df_2016.copy()
    test_df_2016_10['transactiondate'] = pd.Timestamp('2016-10-01')
    test_df_2016_10 = add_date_features(test_df_2016_10)

    test_df_2017_10 = test_df_2017.copy()
    test_df_2017_10['transactiondate'] = pd.Timestamp('2017-10-01')
    test_df_2017_10 = add_date_features(test_df_2017_10)

    X_test_2016_10 = test_df_2016_10[train_features]
    X_test_2017_10 = test_df_2017_10[train_features]

    num_ensembles = 5
    y_pred_2016_10 = 0.0
    y_pred_2017_10 = 0.0

    models = []
    for i in tqdm(range(num_ensembles)):
        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=i)
        model.fit(
            X_train, y_train,
            cat_features=cat_feature_inds)
        models.append(model)
        y_pred_2016_10 += model.predict(X_test_2016_10)
        y_pred_2017_10 += model.predict(X_test_2017_10)

    y_pred_2016_10 /= num_ensembles
    y_pred_2017_10 /= num_ensembles

    submission = pd.DataFrame({
        'ParcelId': test_df_2016['ParcelId'],
    })
    submission['201610'] = y_pred_2016_10
    submission['201611'] = y_pred_2016_10
    submission['201612'] = y_pred_2016_10
    submission['201710'] = y_pred_2017_10
    submission['201711'] = y_pred_2017_10
    submission['201712'] = y_pred_2017_10

    submission.to_csv('data/only_catboost_v5.csv.gz', float_format='%.7f', index=False, compression='gzip')


def main_fun_v3():
    print('Loading Properties ...')
    properties2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    properties2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)

    print('Loading Train ...')
    train2016 = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
    train2017 = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

    def add_date_features(df):
        df["transaction_year"] = df["transactiondate"].dt.year
        df["transaction_month"] = df["transactiondate"].dt.month
        df["transaction_month"] = df["transaction_month"].apply(lambda x: x if x <= 10 else 10)
        # df["transaction_year_month"] = df["transaction_year"] * 100 + df["transaction_month"]
        df["transaction_quarter"] = df["transactiondate"].dt.quarter
        # df["transaction_year_quarter"] = df["transaction_year"] * 10 + df["transaction_quarter"]
        df.drop(["transactiondate"], inplace=True, axis=1)
        return df

    def proc_census_block(prop_data):
        prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
        prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('temp', axis=1, inplace=True)
        prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

        # census_block
        prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
        prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        prop_data.drop('censustractandblock', axis=1, inplace=True)
        return prop_data

    train2016 = add_date_features(train2016)
    train2017 = add_date_features(train2017)

    properties2016 = proc_census_block(properties2016)
    properties2017 = proc_census_block(properties2017)

    print('Loading Sample ...')
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)

    print('Merge Train with Properties ...')
    train2016 = pd.merge(train2016, properties2016, how='left', on='parcelid')
    train2017 = pd.merge(train2017, properties2017, how='left', on='parcelid')

    # print('Tax Features 2017  ...')
    # train2017.iloc[:, train2017.columns.str.startswith('tax')] = np.nan

    print('Concat Train 2016 & 2017 ...')
    train_df = pd.concat([train2016, train2017], axis=0)
    test_df_2016 = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')
    test_df_2017 = pd.merge(sample_submission[['ParcelId']], properties2017.rename(columns={'parcelid': 'ParcelId'}), how='left', on='ParcelId')

    del properties2016, properties2017, train2016, train2017
    gc.collect()

    print('Remove missing data fields ...')

    missing_perc_thresh = 0.98
    exclude_missing = []
    num_rows = train_df.shape[0]
    for c in train_df.columns:
        num_missing = train_df[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(num_rows)
        if missing_frac > missing_perc_thresh:
            exclude_missing.append(c)
    print("We exclude: %s" % len(exclude_missing))

    del num_rows, missing_perc_thresh
    gc.collect()

    print("Remove features with one unique value !!")
    exclude_unique = []
    for c in train_df.columns:
        num_uniques = len(train_df[c].unique())
        if train_df[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            exclude_unique.append(c)
    print("We exclude: %s" % len(exclude_unique))

    print("Define training features !!")
    exclude_other = ['parcelid', 'logerror', 'propertyzoningdesc']
    train_features = []
    for c in train_df.columns:
        if c not in exclude_missing \
                and c not in exclude_other and c not in exclude_unique:
            train_features.append(c)
    print("We use these for training: %s" % len(train_features))

    print("Define categorial features !!")
    cat_feature_inds = []
    cat_unique_thresh = 1000
    for i, c in enumerate(train_features):
        if c in {'census', 'block', 'raw_census', 'raw_block'}:
            cat_feature_inds.append(i)
        else:
            num_uniques = len(train_df[c].unique())
            if num_uniques < cat_unique_thresh \
                    and not 'sqft' in c \
                    and not 'cnt' in c \
                    and not 'nbr' in c \
                    and not 'number' in c:
                cat_feature_inds.append(i)

    print("Cat features are: %s" % [train_features[ind] for ind in cat_feature_inds])
    print("all train features are: %s" % ','.join(train_features))

    print("Replacing NaN values by -999 !!")
    train_df.fillna(-999, inplace=True)
    test_df_2016.fillna(-999, inplace=True)
    test_df_2017.fillna(-999, inplace=True)

    print("Training time !!")
    X_train = train_df[train_features]
    # pkl.dump(X_train, open('only_catboost_x_train_v2.pkl', 'wb'))
    y_train = train_df.logerror
    # pkl.dump(y_train, open('only_catboost_y_train_v2.pkl', 'wb'))
    print(X_train.shape, y_train.shape)

    test_df_2016_10 = test_df_2016.copy()
    test_df_2016_10['transactiondate'] = pd.Timestamp('2016-10-01')
    test_df_2016_10 = add_date_features(test_df_2016_10)

    test_df_2017_10 = test_df_2017.copy()
    test_df_2017_10['transactiondate'] = pd.Timestamp('2017-10-01')
    test_df_2017_10 = add_date_features(test_df_2017_10)

    X_test_2016_10 = test_df_2016_10[train_features]
    X_test_2017_10 = test_df_2017_10[train_features]

    num_ensembles = 3
    y_pred_2016_10 = 0.0
    y_pred_2017_10 = 0.0

    models = []
    for i in tqdm(range(num_ensembles)):
        model = CatBoostRegressor(
            iterations=630, learning_rate=0.03,
            depth=6, l2_leaf_reg=3,
            loss_function='MAE',
            eval_metric='MAE',
            random_seed=i)
        model.fit(
            X_train, y_train,
            cat_features=cat_feature_inds)
        models.append(model)
        y_pred_2016_10 += model.predict(X_test_2016_10)
        y_pred_2017_10 += model.predict(X_test_2017_10)

    y_pred_2016_10 /= num_ensembles
    y_pred_2017_10 /= num_ensembles

    submission = pd.DataFrame({
        'ParcelId': test_df_2016['ParcelId'],
    })
    submission['201610'] = y_pred_2016_10
    submission['201611'] = y_pred_2016_10
    submission['201612'] = y_pred_2016_10
    submission['201710'] = y_pred_2017_10
    submission['201711'] = y_pred_2017_10
    submission['201712'] = y_pred_2017_10

    submission.to_csv('data/only_catboost_v5.csv.gz', float_format='%.7f', index=False, compression='gzip')
