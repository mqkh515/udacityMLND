import numpy as np
import pandas as pd
import data_prep
import models


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')


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
        data_prep.cat_num_to_str(data, new_col_name)

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


def feature_importance(gbm, label=None, print_to_scr=True):
    """only applicable for niave gbm"""
    features = np.array(gbm.feature_name())
    n_features = features.shape[0]

    importance_split = np.array(gbm.feature_importance('split'))
    rank_split = np.argsort(np.argsort(importance_split))
    rank_split = n_features - rank_split
    importance_gain = np.array(gbm.feature_importance('gain'))
    rank_gain = np.argsort(np.argsort(importance_gain))
    rank_gain = n_features - rank_gain
    avg_rank = (rank_gain + rank_split) / 2
    rank_sort_idx = np.argsort(avg_rank)
    rank_sorted = avg_rank[rank_sort_idx]
    features_sorted = features[rank_sort_idx]

    importance_split = gbm.feature_importance('split')
    sort_split = np.argsort(importance_split)
    sort_split = sort_split[::-1]
    features_split_sort = features[sort_split]
    features_class_split = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_split_sort]
    importance_split_sort = importance_split[sort_split]

    importance_gain = gbm.feature_importance('gain')
    sort_gain = np.argsort(importance_gain)
    sort_gain = sort_gain[::-1]
    features_gain_sort = features[sort_gain]
    features_class_gain = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_gain_sort]
    importance_gain_sort = importance_gain[sort_gain]

    df_display = pd.DataFrame()
    df_display['features_avg'] = features_sorted
    df_display['avg_rank'] = rank_sorted
    df_display['class_avg'] = [feature_info.loc[f, 'class'] if f in feature_info.index else 'new' for f in features_sorted]
    df_display['split_rank_avg'] = rank_split[rank_sort_idx]
    df_display['gain_rank_avg'] = rank_gain[rank_sort_idx]
    df_display['feature_split'] = features_split_sort
    df_display['class_split'] = features_class_split
    df_display['split'] = importance_split_sort
    df_display['feature_gain'] = features_gain_sort
    df_display['class_gain'] = features_class_gain
    df_display['gain'] = importance_gain_sort

    df_display = df_display[['features_avg', 'class_avg', 'avg_rank', 'split_rank_avg', 'gain_rank_avg',
                             'feature_split', 'class_split', 'split', 'feature_gain', 'class_gain', 'gain']]
    if label:
        df_display.to_csv('records/feature_importance_%s.csv' % label, index=False)
    if print_to_scr:
        print(df_display)
    return features_sorted, rank_sorted


def class_3_feature_search():
    """find the best group-mean delegate for class-3 features"""
    # create_group_mean, group_count variables
    num_vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
                'dollar_taxvalue_structure_land_diff_norm', 'dollar_taxvalue_structure_land_absdiff_norm',
                'dollar_taxvalue_structure_total_ratio', 'dollar_taxvalue_total_dollar_tax_ratio')
    picked_vars = []

    for var_class3 in ('block',
                       'census',
                       'code_county_landuse',
                       'str_zoning_desc',
                       'raw_block',
                       'raw_census',
                       'code_city',
                       'code_neighborhood',
                       'code_zip'):

        new_features = []
        for num_var in num_vars:
            if num_var == num_vars[0]:
                new_features.append(num_var + '__groupby__' + var_class3 + '__count')
            new_features.append(num_var + '__groupby__' + var_class3 + '__mean')
        new_features.append(var_class3 + '_lgb')

        prop_2016, prop_2017 = data_prep.prop_2016.copy(), data_prep.prop_2017.copy()
        for f in new_features:
            feature_factory(f, prop_2016)
            feature_factory(f, prop_2017)

        train_x, train_y = data_prep.load_train_data(prop_2016, prop_2017, recalc=True)
        model_lgb_raw = models.ModelLGBRaw()
        model_lgb_raw.added_features = new_features
        model_lgb_raw.train(train_x, train_y, dump_model=False)
        feature_list, feature_rank = feature_importance(model_lgb_raw.model, None, False)

        idx = np.array([True if f in new_features else False for f in feature_list])
        out_df = pd.DataFrame({'rank': feature_rank[idx]}, index=feature_list[idx])
        out_df.to_csv('class3_search/%s.csv' % var_class3, index=False)
        picked_var_arg = np.argmin(out_df['rank'])
        picked_vars.append([var_class3, out_df['rank'][var_class3 + '_lgb'], picked_var_arg, out_df['rank'][picked_var_arg]])

    out_df_all = pd.DataFrame(picked_vars, columns=['raw_var', 'raw_var_rank', 'picked_var', 'picked_var_rank'])
    out_df_all.to_csv('class3_search/picked_vars.csv', index=False)
