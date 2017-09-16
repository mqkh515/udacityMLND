import numpy as np
import pandas as pd
import math
import gc
from time import time
import matplotlib.pyplot as plt

# init load data
prop_data_raw = pd.read_csv('data/properties_2016.csv', header=0)
error_data = pd.read_csv('data/train_2016_v2.csv', header=0)
error_data['sale_month'] = error_data['transactiondate'].apply(lambda x: x.split('-')[1])  # get error data transaction date to month

train_data_raw = error_data.merge(prop_data_raw, how='left', on='parcelid')
train_data_raw.to_csv('data/train_data_merge.csv', index=False)

prop_data = prop_data_raw.copy()
train_data = train_data_raw.copy()


def cat_num_to_str(col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not prop_data[col_name].dtype == 'O':
        prop_data[col_name] = prop_data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)
    if not train_data[col_name].dtype == 'O':
        train_data[col_name] = train_data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def mark_flag_col(col_name):
    """mark bool for numerical columns, mark True for val > 0, and False otherwise (include NaN)"""
    prop_data_marks_true = prop_data[col_name] >= 0.5
    prop_data.loc[prop_data.index[prop_data_marks_true], col_name] = 'TRUE'
    prop_data.loc[prop_data.index[~prop_data_marks_true], col_name] = 'FALSE'
    train_data_marks_true = train_data[col_name] >= 0.5
    train_data.loc[train_data.index[train_data_marks_true], col_name] = 'TRUE'
    train_data.loc[train_data.index[~train_data_marks_true], col_name] = 'FALSE'


def mark_flag_col_tax_delinquency():
    prop_data_marks_true = prop_data['taxdelinquencyflag'] == 'Y'
    prop_data.loc[prop_data.index[prop_data_marks_true], 'taxdelinquencyflag'] = 'TRUE'
    prop_data.loc[prop_data.index[~prop_data_marks_true], 'taxdelinquencyflag'] = 'FALSE'
    train_data_marks_true = train_data['taxdelinquencyflag'] == 'Y'
    train_data.loc[train_data.index[train_data_marks_true], 'taxdelinquencyflag'] = 'TRUE'
    train_data.loc[train_data.index[~train_data_marks_true], 'taxdelinquencyflag'] = 'FALSE'


def col_fill_na(data, col, fill):
    if fill == 'mode':
        data[col].fillna(data[col].mode().values[0], inplace=True)
    elif fill == 'mean':
        data[col].fillna(data[col].mean(), inplace=True)
    elif fill == '0':
        data[col].fillna(0, inplace=True)
    else:
        print('unknown fill type: %s' % fill)


def clear_cat_col_group(data, col, groups):
    """set given groups of a categorical col to na"""
    in_group_flag = data[col].apply(lambda x: x in groups)
    data[col].loc[data[col].index[in_group_flag]] = np.nan


# special treatment functions
def raw_census_info_split():
    # create new categorical columns 'raw_census', 'raw_block'
    def raw_census_info_split_inner(data):
        data['temp'] = data['rawcensustractandblock'].apply(
            lambda x: str(round(x * 1000000)) if not np.isnan(x) else 'nan')
        data['temp'] = data['temp'].astype('O')
        data['raw_census'] = data['temp'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        data['raw_block'] = data['temp'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)
        data.drop('temp', axis=1, inplace=True)

    raw_census_info_split_inner(prop_data)
    raw_census_info_split_inner(train_data)


def census_info_split():
    # create new categorical columns 'raw_census', 'raw_block'
    def census_info_split_inner(data):
        data['census'] = data['censustractandblock'].apply(lambda x: x[4:10] if not x == 'nan' else np.nan)
        data['block'] = data['censustractandblock'].apply(lambda x: x[10:] if not x == 'nan' else np.nan)

    prop_data['censustractandblock'] = prop_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
    train_data['censustractandblock'] = train_data['censustractandblock'].apply(lambda x: str(int(x)) if not np.isnan(x) else 'nan')
    census_info_split_inner(prop_data)
    census_info_split_inner(train_data)


def nan_impute_fullbathcnt():
    """nan impute fullbathcnt from bathroomcnt(which is no missing)"""

    def nan_impute_fullbathcnt_inner(data):
        null_idx = data.index[data['fullbathcnt'].isnull()]
        fill_val = data['bathroomcnt'][null_idx].copy()
        fill_val_raw = fill_val.copy()
        fill_val_raw_floor = fill_val_raw.apply(math.floor)
        int_idx = np.abs(fill_val_raw.values - fill_val_raw_floor.values) < 1e-12
        fill_val[int_idx] = np.maximum(fill_val_raw[int_idx] - 1, 0)
        fill_val[~int_idx] = fill_val_raw_floor[~int_idx]
        data.loc[null_idx, 'fullbathcnt'] = fill_val
        return data

    nan_impute_fullbathcnt_inner(prop_data)
    nan_impute_fullbathcnt_inner(train_data)


def garage_area_cleaning():
    """clearn garage features, with zero area but sample has non-zero count"""

    def zero_impute_area_garage_inner(data, data_name):
        nan_idx = data.index[np.logical_and(np.abs(data['garagetotalsqft'] - 0) < 1e-12, data['garagecarcnt'] > 0)]
        data.loc[nan_idx, 'garagetotalsqft'] = np.nan  # to do better to impute with cnt group mean
        print('cleaned rows for %s: %d' % data_name, len(nan_idx))

    zero_impute_area_garage_inner(prop_data, 'prop_data')
    zero_impute_area_garage_inner(train_data, 'train_data')


def property_cleaning_base():
    """basic feature clearning"""
    # type_air_conditioning
    cat_num_to_str('airconditioningtypeid')
    clear_cat_col_group(prop_data, 'airconditioningtypeid', ['12'])

    # type_architectural_style
    cat_num_to_str('architecturalstyletypeid')
    clear_cat_col_group(prop_data, 'architecturalstyletypeid', ['27', '5'])

    # area_base_finished

    # num_bathroom_assessor
    col_fill_na(prop_data, 'bathroomcnt', 'mode')

    # num_bathroom_zillow

    # num_bedroom
    col_fill_na(prop_data, 'bedroomcnt', 'mode')

    # type_building_framing
    cat_num_to_str('buildingclasstypeid')
    clear_cat_col_group(prop_data, 'buildingclasstypeid', ['4'])

    # rank_building_quality

    # type_deck
    cat_num_to_str('decktypeid')

    # area_firstfloor_zillow

    # area_living_finished_calc

    # area_living_type_12

    # area_living_type_13

    # area_living_type_15

    # area_firstfloor_assessor

    # area_living_type_6

    # code_ips, no missing in train
    cat_num_to_str('fips')
    col_fill_na(prop_data, 'fips', 'mode')

    # num_fireplace
    # col_fill_na(prop_data, 'fireplacecnt', '0')

    # num_fullbath
    # nan_impute_fullbathcnt()

    # num_garage

    # area_garage

    # flag_spa_zillow
    mark_flag_col('hashottuborspa')

    # type_heating_system
    cat_num_to_str('heatingorsystemtypeid')
    clear_cat_col_group(prop_data, 'heatingorsystemtypeid', ['19', '21'])

    # latitude
    col_fill_na(prop_data, 'latitude', 'mean')

    # longitude
    col_fill_na(prop_data, 'longitude', 'mean')

    # area_lot

    # flag_pool
    mark_flag_col('poolcnt')

    # area_pool
    prop_data.loc[prop_data.index[prop_data['poolcnt'] == 'FALSE'], 'poolsizesum'] = 0
    train_data.loc[train_data.index[train_data['poolcnt'] == 'FALSE'], 'poolsizesum'] = 0

    # pooltypeid10, high missing rate, counter intuitive values. drop it

    # pooltypeid2 and pooltypeid7
    mark_flag_col('pooltypeid2')
    mark_flag_col('pooltypeid7')

    def make_pool_type(data):
        data['type_pool'] = 'None'
        data.loc[data.index[data['pooltypeid2'] == 'TRUE'], 'type_pool'] = 'TRUE'
        data.loc[data.index[data['pooltypeid7'] == 'TRUE'], 'type_pool'] = 'FALSE'

    make_pool_type(prop_data)
    make_pool_type(train_data)

    # code_county_landuse
    col_fill_na(prop_data, 'propertycountylandusecode', 'mode')

    # code_county_landuse
    cat_num_to_str('propertylandusetypeid')
    col_fill_na(prop_data, 'propertylandusetypeid', 'mode')
    clear_cat_col_group(prop_data, 'propertylandusetypeid', ['270'])

    # str_zoning_desc

    # raw_census_block, raw_census, raw_block.
    raw_census_info_split()
    col_fill_na(prop_data, 'raw_census', 'mode')
    col_fill_na(prop_data, 'raw_block', 'mode')

    # code_city
    cat_num_to_str('regionidcity')

    # code_county
    cat_num_to_str('regionidcounty')

    # code_neighborhood
    cat_num_to_str('regionidneighborhood')

    # code_zip
    cat_num_to_str('regionidzip')

    # num_room
    col_fill_na(prop_data, 'roomcnt', 'mode')

    # type_story
    cat_num_to_str('storytypeid')

    # num_34_bathroom

    # type_construction
    clear_cat_col_group(prop_data, 'typeconstructiontypeid', ['2'])
    cat_num_to_str('typeconstructiontypeid')

    # num_unit

    # area_yard_patio

    # area_yard_storage

    # year_built

    # num_story
    cat_num_to_str('numberofstories')

    # flag_fireplace
    mark_flag_col('fireplaceflag')

    # dollar_taxvalue_structure

    # dollar_taxvalue_total
    col_fill_na(prop_data, 'taxvaluedollarcnt', 'mean')

    # dollar_taxvalue_land
    col_fill_na(prop_data, 'landtaxvaluedollarcnt', 'mean')

    # dollar_tax

    # flag_tax_delinquency
    mark_flag_col_tax_delinquency()

    # year_tax_due

    # census_block


def property_engineering():
    """for real kaggle submission, considering:
       1, outlier detection.
       2, missing data imputation from prop data.
       3, categorical variables group combining."""
    pass


def property_imputation(col_name):
    pass


def multi_trade_analysis(error_data_inp):
    """an asset can be traded more than once within a year, can multi_trade related to large error? should we exclude these samples from training?"""
    error_data = error_data_inp.copy()
    print('average abs logerror in all traning transactions: %4.4f' % np.abs(error_data['logerror']).mean())  # 0.0684
    n_trade = error_data['logerror'].groupby(error_data['parcelid']).count()
    multi_trade_count = n_trade[n_trade > 1]
    print('number of parcels traded more than once in training data: %d' % len(multi_trade_count))  # 124, number too small
    multi_trade_parcel_set = set(multi_trade_count.index.values)
    error_data['is_multi_trade'] = error_data['parcelid'].apply(lambda x: x in multi_trade_parcel_set)
    error_data_multi_trade = error_data[error_data['is_multi_trade']]
    print('average abs logerror in all multi_trade transactions: %4.4f' % np.abs(error_data_multi_trade['logerror']).mean())  # 0.0921




