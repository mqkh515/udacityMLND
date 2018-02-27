import numpy as np
import pandas as pd
import data_prep
import models


feature_info = pd.read_csv('data/feature_info.csv', index_col='new_name')


def feature_importance(gbm):
    """only applicable for niave gbm"""

    def get_feature_class(f):
        f = f[:-4] if f[-4:] == '_lgb' else f
        return feature_info.loc[f, 'class'] if f in feature_info.index else 'new'

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
    features_class_split = [get_feature_class(f) for f in features_split_sort]
    importance_split_sort = importance_split[sort_split]

    importance_gain = gbm.feature_importance('gain')
    sort_gain = np.argsort(importance_gain)
    sort_gain = sort_gain[::-1]
    features_gain_sort = features[sort_gain]
    features_class_gain = [get_feature_class(f) for f in features_gain_sort]
    importance_gain_sort = importance_gain[sort_gain]

    df_display = pd.DataFrame()
    df_display['features'] = features_sorted
    df_display['avg_rank'] = rank_sorted
    df_display['class'] = [get_feature_class(f) for f in features_sorted]
    df_display['split_rank'] = rank_split[rank_sort_idx]
    df_display['gain_rank'] = rank_gain[rank_sort_idx]
    df_display['feature_split'] = features_split_sort
    df_display['class_split'] = features_class_split
    df_display['split'] = importance_split_sort
    df_display['feature_gain'] = features_gain_sort
    df_display['class_gain'] = features_class_gain
    df_display['gain'] = importance_gain_sort

    df_display = df_display[['features', 'class', 'avg_rank', 'split_rank', 'gain_rank',
                             'feature_split', 'class_split', 'split', 'feature_gain', 'class_gain', 'gain']]

    return df_display


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
            data_prep.feature_factory(f, prop_2016)
            data_prep.feature_factory(f, prop_2017)

        train_x, train_y = data_prep.load_train_data(prop_2016, prop_2017, recalc=True)
        model_lgb = models.ModelLGBRawIncMonOutlierRm()
        model_lgb.added_features += new_features
        model_lgb.train(train_x, train_y, dump_model=False)
        feature_info = feature_importance(model_lgb.model)
        feature_list, feature_rank = feature_info['features'], feature_info['avg_rank']

        idx = np.array([True if f in new_features else False for f in feature_list])
        out_df = pd.DataFrame({'rank': feature_rank[idx].values}, index=feature_list[idx].values)
        out_df.to_csv('class3_search/%s.csv' % var_class3, index=False)
        picked_var_arg = np.argmin(out_df['rank'])
        picked_vars.append([var_class3, out_df['rank'][var_class3 + '_lgb'], picked_var_arg, out_df['rank'][picked_var_arg]])

    out_df_all = pd.DataFrame(picked_vars, columns=['raw_var', 'raw_var_rank', 'picked_var', 'picked_var_rank'])
    out_df_all.to_csv('class3_search/picked_vars.csv', index=False)
