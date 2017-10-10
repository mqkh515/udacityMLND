import pandas as pd
import numpy as np


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')
nmap_orig_to_new =  dict(zip(feature_info['orig_name'].values, feature_info.index.values))
nmap_new_to_orig =  dict(zip(feature_info.index.values, feature_info['orig_name'].values))


def cat_num_to_str(data, col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not data[col_name].dtype == 'O':
        data.loc[:, col_name] = data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def prep_for_lgb_single(data):
    """map categorical variables to int for lgb run"""
    for col in data.columns:
        if col in feature_info.index and feature_info.loc[col, 'type'] == 'cat':
            convert_cat_col_single(data, col)


def convert_cat_col_single(data, col):
    # convert to lgb usable categorical
    # set nan to string so that can be sorted, make sure training set and testing set get the same coding
    data.loc[data.index[data[col].isnull()], col] = 'nan'
    uni_vals = np.sort(data[col].unique()).tolist()
    map = dict(zip(uni_vals, list(range(1, len(uni_vals) + 1))))
    col_new = col + '_lgb'
    data[col_new] = data[col].apply(lambda x: map[x])
    data[col_new] = data[col_new].astype('category')


def property_trans(prop_data):
    """only does transformation, no cleaning"""

    def mark_flag_col(col_name):
        """mark bool for numerical columns, mark True for val > 0, and False otherwise (include NaN)"""
        data_marks_true = prop_data[col_name] >= 0.5
        prop_data.loc[prop_data.index[data_marks_true], col_name] = 'TRUE'
        prop_data.loc[prop_data.index[~data_marks_true], col_name] = 'FALSE'

    def mark_flag_col_tax_delinquency():
        test_data_marks_true = prop_data['taxdelinquencyflag'] == 'Y'
        prop_data.loc[prop_data.index[test_data_marks_true], 'taxdelinquencyflag'] = 'TRUE'
        prop_data.loc[prop_data.index[~test_data_marks_true], 'taxdelinquencyflag'] = 'FALSE'

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
    prop_data['temp'] = prop_data['rawcensustractandblock'].apply(lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
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

    # census_block
    prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
    prop_data['census'] = prop_data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
    prop_data['block'] = prop_data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
    prop_data.drop('censustractandblock', axis=1, inplace=True)

    # for name in feature_info.index.values:
    #     if name[:4] == 'num_':
    #         cat_num_to_str(prop_data, nmap_new_to_orig[name])
    #         feature_info.loc[name, 'type'] = 'cat'

    # name columns
    prop_data.rename(columns=nmap_orig_to_new, inplace=True)


def property_cleaning_v2(prop_data):
    """clean cols according to train_data distribution"""

    def clear_cat_col_group(col, groups):
        """set given groups of a categorical col to na"""
        in_group_flag = prop_data[col].apply(lambda x: x in groups)
        prop_data.loc[prop_data[col].index[in_group_flag], col] = np.nan

    clear_cat_col_group('type_air_conditioning', ['12'])
    clear_cat_col_group('type_architectural_style', ['27', '5'])
    clear_cat_col_group('type_heating_system', ['19', '21'])
    clear_cat_col_group('type_construction', ['11'])
    clear_cat_col_group('type_landuse', ['270', '279'])


    # num_garage and area_garage not consistent
    mark_idx = prop_data.index[np.logical_and(np.abs(prop_data['area_garage'] - 0) < 1e-12, prop_data['num_garage'] > 0)]
    sub_df = prop_data.loc[mark_idx, ['area_garage', 'num_garage']]
    sub_df = sub_df.join(prop_data['area_garage'].groupby(prop_data['num_garage']).mean(), on='num_garage', rsuffix='_avg')
    prop_data.loc[mark_idx, 'area_garage'] = sub_df['area_garage_avg']


def load_prop_data():
    sample_submission = pd.read_csv('data/sample_submission.csv', low_memory=False)
    sub_index = sample_submission['ParcelId']
    prop_2016 = pd.read_csv('data/properties_2016.csv', low_memory=False)
    prop_2016.index = prop_2016['parcelid']
    prop_2017 = pd.read_csv('data/properties_2017.csv', low_memory=False)
    prop_2017.index = prop_2017['parcelid']

    def size_control(data):
        data.loc[:, ['latitude', 'longitude']] = data.loc[:, ['latitude', 'longitude']] / 1e6
        for col in data.columns:
            if data[col].dtype == np.float64:
                data.loc[:, col] = data[col].astype(np.float32)

    size_control(prop_2016)
    size_control(prop_2017)

    # make sure prediction index order is inline with expected submission index order
    prop_2016 = prop_2016.loc[sub_index, :]
    prop_2017 = prop_2017.loc[sub_index, :]

    property_trans(prop_2016)
    property_trans(prop_2017)

    property_cleaning_v2(prop_2016)
    property_cleaning_v2(prop_2017)

    n_row_prop_2016 = prop_2016.shape[0]
    n_row_prop_2017 = prop_2017.shape[0]
    prop_join = pd.concat([prop_2016, prop_2017], axis=0)

    prep_for_lgb_single(prop_join)
    prop_2016 = prop_join.iloc[list(range(n_row_prop_2016)), :]
    prop_2017 = prop_join.iloc[list(range(n_row_prop_2016, n_row_prop_2016 + n_row_prop_2017)), :]

    return prop_2016, prop_2017


prop_2016, prop_2017 = load_prop_data()