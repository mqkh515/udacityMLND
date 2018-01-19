import zillow_1 as z
import params_cache as p
import pandas as pd
import numpy as np
import os

if not os.path.exists('class3_research'):
    os.mkdir('class3_research')


def class_3_var_rank():

    # create_group_mean, group_count variables
    num_vars = ('dollar_tax', 'dollar_taxvalue_structure', 'dollar_taxvalue_land', 'dollar_taxvalue_total',
                'dollar_taxvalue_structure_land_diff_norm', 'dollar_taxvalue_structure_land_absdiff_norm',
                'dollar_taxvalue_structure_total_ratio', 'dollar_taxvalue_total_dollar_tax_ratio')

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

        train_x, train_y = z.load_train_data(z.prop_2016, z.prop_2017, new_features)
        x_raw, y = z.capfloor_outlier(train_x, train_y)
        x = z.lgb_data_prep(x_raw, new_features)
        gbm = z.train_lgb(x, y, z.get_params(p.raw_lgb_2y_1, 'reg'))
        feature_list, feature_rank = z.feature_importance(gbm, None, False)

        new_features.append(var_class3)
        idx = np.array([True if f in new_features else False for f in feature_list])
        out_df = pd.DataFrame({'feature': feature_list[idx], 'rank': feature_rank[idx]})
        out_df.to_csv('class3_research/%s.csv' % var_class3, index=False)

