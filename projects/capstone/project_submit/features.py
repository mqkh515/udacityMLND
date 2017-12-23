import numpy as np


def cat_num_to_str(data, col_name):
    """for numeric-like categorical varible, transform to string, keep nan"""
    if not data[col_name].dtype == 'O':
        data.loc[:, col_name] = data[col_name].apply(lambda x: str(int(x)) if not np.isnan(x) else np.nan)


def create_type_var(data, col):
    """create type_var from given col.
        1, group or mark small categories as NA.
        2, can also do this for num_ vars, and transform them to cat.
        3, create a new col groupby_col in data"""

    new_col_name = 'groupby__' + col

    if col in ('type_air_conditioning', 'flag_pool', 'flag_tax_delinquency',
               'str_zoning_desc', 'code_city', 'code_neighborhood', 'code_zip',
               'raw_block', 'raw_census', 'block', 'census', 'code_fips'):
        # already grouped in basic cleaning
        data[new_col_name] = data[col].copy()
    else:
        data[new_col_name] = data[col].copy()
        cat_num_to_str(data, new_col_name)

    return new_col_name


def groupby_feature_gen(num_col, group_col, data, raw_data, op_type):
    """data: already applied lgb prep, can be directly used for lgb train.
       raw_data: type_var before lgb prep, used for group_var and mean_var creation.
       test_data: full data, used to determine nan index.
       created feature follows format num_var__groupby__cat_var__op_type"""
    new_cat_var = create_type_var(raw_data, group_col)

    # first find too small groups and remove, too small groups are defined as
    # 1, first sort groups by group size.
    # 2, cumsum <= total non-nan samples * 0.001
    group_count = raw_data[new_cat_var].groupby(raw_data[new_cat_var]).count()
    group_count = group_count.sort_values()
    small_group = set(group_count.index[group_count.cumsum() < int(np.sum(~raw_data[new_cat_var].isnull()) * 0.0001)])

    rm_idx = raw_data[new_cat_var].apply(lambda x: x in small_group)

    # now create variables
    # group_avg
    mean_col_name = num_col + '__' + new_cat_var + '__mean'
    if mean_col_name in raw_data.columns:
        avg_col = raw_data[mean_col_name]
    else:
        avg_col = raw_data[new_cat_var].map(raw_data[num_col].groupby(raw_data[new_cat_var]).mean())
    # neu col
    if op_type == 'neu':
        new_col = num_col + '__' + new_cat_var + '__neu'
        data[new_col] = (raw_data[num_col] - avg_col) / avg_col
        data.loc[avg_col < 1e-5, new_col] = np.nan
    elif op_type == 'absneu':
        new_col = num_col + '__' + new_cat_var + '__absneu'
        data[new_col] = np.abs(raw_data[num_col] - avg_col) / avg_col
        data.loc[avg_col < 1e-5, new_col] = np.nan
    elif op_type == 'mean':
        new_col = num_col + '__' + new_cat_var + '__mean'
        data[new_col] = avg_col
    elif op_type == 'count':
        new_col = num_col + '__' + new_cat_var + '__count'
        data[new_col] = raw_data[new_cat_var].groupby(raw_data[new_cat_var]).count()
    else:
        raise Exception('unknown op type')

    # rm small group idx
    data.loc[rm_idx, new_col] = np.nan
    # clear newly created col
    raw_data.drop(new_cat_var, axis=1, inplace=True)

    return new_col


def feature_factory(col, data, keep_num_var=False):
    """add col to data, generated from raw data columns"""

    if col == 'dollar_taxvalue_structure_land_diff':
        data[col] = data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land']

    if col == 'dollar_taxvalue_structure_land_absdiff':
        data[col] = np.abs(data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land'])

    if col == 'dollar_taxvalue_structure_land_diff_norm':
        data[col] = (data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land']) / data['dollar_taxvalue_total']

    if col == 'dollar_taxvalue_structure_land_absdiff_norm':
        data[col] = np.abs(data['dollar_taxvalue_structure'] - data['dollar_taxvalue_land']) / data['dollar_taxvalue_total']

    if col == 'dollar_taxvalue_structure_total_ratio':
        data[col] = data['dollar_taxvalue_structure'] / data['dollar_taxvalue_total']

    if col == 'dollar_taxvalue_total_dollar_tax_ratio':
        data[col] = data['dollar_taxvalue_total'] / data['dollar_tax']

    if col == 'living_area_proportion':
        data[col] = data['area_living_type_12'] / data['area_lot']

    if '__per__' in col:
        if not '__groupby__' in col:
            v, a = col.split('__per__')
            data[col] = data[v] / data[a]
            data.loc[np.abs(data[a]) < 1e-5, col] = np.nan

    if '__groupby__' in col:
        num_var, cat_var_and_op_type = col.split('__groupby__')
        cat_var, op_type = cat_var_and_op_type.split('__')
        num_var_in_raw_data = num_var in data.columns
        if not num_var_in_raw_data:
            feature_factory(num_var, data)
        _ = groupby_feature_gen(num_var, cat_var, data, data, op_type)
        if not num_var_in_raw_data:
            if not keep_num_var:
                data.drop(num_var, axis=1, inplace=True)
            else:
                print('num var kept: %s' % num_var)