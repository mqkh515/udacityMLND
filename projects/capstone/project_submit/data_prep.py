import pandas as pd
import numpy as np
import pickle as pkl
import os


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')
nmap_orig_to_new = dict(zip(feature_info['orig_name'].values, feature_info.index.values))
nmap_new_to_orig = dict(zip(feature_info.index.values, feature_info['orig_name'].values))


def cat_num_to_str(data, col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not data[col_name].dtype == 'O':
        data.loc[:, col_name] = data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def property_trans(prop_data):
    """only does transformation, no cleaning"""

    def mark_flag_col(col_name):
        """mark bool for numerical columns, mark True for val > 0, and False otherwise (include NaN)"""
        data_marks_true = prop_data[col_name] >= 0.5
        prop_data.loc[prop_data.index[data_marks_true], col_name] = 'TRUE'
        prop_data.loc[prop_data.index[~data_marks_true], col_name] = 'FALSE'

    cat_num_to_str(prop_data, 'airconditioningtypeid')
    cat_num_to_str(prop_data, 'architecturalstyletypeid')
    cat_num_to_str(prop_data, 'buildingclasstypeid')
    cat_num_to_str(prop_data, 'decktypeid')
    cat_num_to_str(prop_data, 'fips')
    mark_flag_col('hashottuborspa')
    cat_num_to_str(prop_data, 'heatingorsystemtypeid')
    mark_flag_col('poolcnt')
    mark_flag_col('pooltypeid10')
    mark_flag_col('pooltypeid2')
    mark_flag_col('pooltypeid7')
    prop_data['type_pool'] = 'None'
    prop_data.loc[prop_data.index[prop_data['pooltypeid2'] == 'TRUE'], 'type_pool'] = 'TRUE'
    prop_data.loc[prop_data.index[prop_data['pooltypeid7'] == 'TRUE'], 'type_pool'] = 'FALSE'
    cat_num_to_str(prop_data, 'propertycountylandusecode')
    cat_num_to_str(prop_data, 'propertylandusetypeid')
    cat_num_to_str(prop_data, 'propertyzoningdesc')

    # raw_census_block, raw_census, raw_block.
    prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(int(round(x * 1000000))) if not np.isnan(x) else 'nan')
    prop_data['raw_census'] = prop_data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['raw_block'] = prop_data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('temp', axis=1, inplace=True)
    prop_data.drop('rawcensustractandblock', axis=1, inplace=True)

    cat_num_to_str(prop_data, 'regionidcity')
    cat_num_to_str(prop_data, 'regionidcounty')
    cat_num_to_str(prop_data, 'regionidneighborhood')
    cat_num_to_str(prop_data, 'regionidzip')
    cat_num_to_str(prop_data, 'storytypeid')
    cat_num_to_str(prop_data, 'typeconstructiontypeid')
    mark_flag_col('fireplaceflag')

    prop_data.loc[prop_data.index[prop_data['taxdelinquencyflag'] == 'Y'], 'taxdelinquencyflag'] = 'TRUE'
    prop_data.loc[prop_data.index[~(prop_data['taxdelinquencyflag'] == 'Y')], 'taxdelinquencyflag'] = 'FALSE'

    # census_block
    prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
    prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('censustractandblock', axis=1, inplace=True)

    prop_data.rename(columns=nmap_orig_to_new, inplace=True)


def prep_for_lgb(data):
    """map categorical variables to int for lgb run"""
    for col in data.columns:
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'cat':
            data.loc[data.index[data[col].isnull()], col] = 'nan'
            try:
                uni_vals = np.sort(data[col].unique()).tolist()
            except:
                print('sort exception, col processing: %s' % col)
                continue
            map = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
            col_new = col + '_lgb'
            data[col_new] = data[col].apply(lambda x: map[x])
            data[col_new] = data[col_new].astype('category')


def property_cleaning(prop_data):
    """clean cols according to train_data distribution"""

    def clear_cat_col_group(col, groups):
        """set given groups of a categorical col to na"""
        in_group_flag = prop_data[col].apply(lambda x: x in groups)
        prop_data.loc[prop_data[col].index[in_group_flag], col] = np.nan

    def col_fill_na(col, fill_type):
        if fill_type == 'mode':
            prop_data[col].fillna(prop_data[col].mode().values[0], inplace=True)
        elif fill_type == 'mean':
            prop_data[col].fillna(prop_data[col].mean(), inplace=True)
        elif fill_type == '0':
            prop_data[col].fillna(0, inplace=True)
        else:
            raise Exception('unknown fill_type')

    clear_cat_col_group('type_air_conditioning', ['12'])
    clear_cat_col_group('type_architectural_style', ['27', '5'])
    clear_cat_col_group('type_building_framing', ['1', '2', '5'])
    clear_cat_col_group('type_heating_system', ['19', '21'])
    clear_cat_col_group('type_construction', ['11'])
    clear_cat_col_group('type_landuse', ['270', '279'])

    # num_garage and area_garage not consistent
    mark_idx = prop_data.index[np.logical_and(np.abs(prop_data['area_garage'] - 0) < 1e-12, prop_data['num_garage'] > 0)]
    sub_df = prop_data.loc[mark_idx, ['area_garage', 'num_garage']]
    sub_df = sub_df.join(prop_data['area_garage'].groupby(prop_data['num_garage']).median(), on='num_garage', rsuffix='_avg')
    prop_data.loc[mark_idx, 'area_garage'] = sub_df['area_garage_avg']

    # area_pool, mark area as 0 for pool count as False
    prop_data.loc[prop_data.index[prop_data['flag_pool'] == 'FALSE'], 'area_pool'] = 0

    # num_fireplace (if any)
    col_fill_na('num_fireplace', '0')

    # num_34_batchroom, min value is 1, use 0 for na
    col_fill_na('num_34_bathroom', '0')

    # area_pool to be made consistent with flag_pool
    prop_data.loc[prop_data.index[prop_data['flag_pool'] == 'FALSE'], 'area_pool'] = 0


def create_prop_data():
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)
    sub_index = sample_submission['ParcelId']
    prop_2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    prop_2016.index = prop_2016['parcelid']
    prop_2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)
    prop_2017.index = prop_2017['parcelid']

    # make sure prediction index order is inline with expected submission index order
    prop_2016 = prop_2016.loc[sub_index, :]
    prop_2017 = prop_2017.loc[sub_index, :]

    property_trans(prop_2016)
    property_trans(prop_2017)

    property_cleaning(prop_2016)
    property_cleaning(prop_2017)

    # combine 2016 and 2017 data, create consistent coding for categorical variables
    n_row_prop_2016 = prop_2016.shape[0]
    n_row_prop_2017 = prop_2017.shape[0]
    prop_join = pd.concat([prop_2016, prop_2017], axis=0)
    prep_for_lgb(prop_join)
    prop_2016 = prop_join.iloc[list(range(n_row_prop_2016)), :]
    prop_2017 = prop_join.iloc[list(range(n_row_prop_2016, n_row_prop_2016 + n_row_prop_2017)), :]

    return prop_2016, prop_2017


def load_prop_data():
    if not os.path.exists('data/prop_2016.pkl') or not os.path.exists('data/prop_2017.pkl'):
        prop_2016, prop_2017 = create_prop_data()
        pkl.dump(prop_2016, open('data/prop_2016.pkl', 'wb'))
        pkl.dump(prop_2017, open('data/prop_2017.pkl', 'wb'))
    else:
        prop_2016 = pkl.load(open('data/prop_2016.pkl', 'rb'))
        prop_2017 = pkl.load(open('data/prop_2017.pkl', 'rb'))
    return prop_2016, prop_2017


# prop_2016, prop_2017 = load_prop_data()
prop_2016, prop_2017 = None, None


def load_train_data(prop_2016, prop_2017, recalc=False, force_dump=False):
    """if engineered features exists, it should be performed at prop_data level, and then join to error data"""
    if not os.path.exists('data/train_x.pkl') or not os.path.exists('data/train_y.pkl') or recalc:
        train_2016 = pd.read_csv('data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
        train_2017 = pd.read_csv('data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

        train_2016 = pd.merge(train_2016, prop_2016, how='left', on='parcelid')
        train_2016['data_year'] = 2016
        train_2017 = pd.merge(train_2017, prop_2017, how='left', on='parcelid')
        train_2017['data_year'] = 2017

        train = pd.concat([train_2016, train_2017], axis=0)
        train.index = list(range(train.shape[0]))

        # sale time variables creation
        train['sale_year'] = train['transactiondate'].dt.year
        train['sale_month'] = train['transactiondate'].dt.month
        train['sale_quarter'] = train['transactiondate'].dt.quarter
        train.drop(['transactiondate'], inplace=True, axis=1)

        train_y = train['logerror']
        train_x = train.drop('logerror', axis=1)
        if force_dump:
            pkl.dump(train_x, open('data/train_x.pkl', 'wb'))
            pkl.dump(train_y, open('data/train_y.pkl', 'wb'))
    else:
        train_x = pkl.load(open('data/train_x.pkl', 'rb'))
        train_y = pkl.load(open('data/train_y.pkl', 'rb'))
    return train_x, train_y


train_x, train_y = load_train_data(prop_2016, prop_2017, False, True)

