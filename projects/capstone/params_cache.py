raw_lgb_2y_1 = {'num_leaves': 77,
                'min_data_in_leaf': 244,
                'learning_rate': 0.005218,
                'num_boosting_rounds': 2300}

raw_lgb_2y_2 = {'num_leaves': 47,
                'min_data_in_leaf': 257,
                'learning_rate': 0.011487,
                'num_boosting_rounds': 1400}

raw_lgb_2y_3 = {'num_leaves': 60,
                'min_data_in_leaf': 349,
                'learning_rate': 0.008397,
                'num_boosting_rounds': 1600}

raw_lgb_2y_4 = {'num_leaves': 77,
                'min_data_in_leaf': 244,
                'learning_rate': 0.005218,
                'num_boosting_rounds': 2500}

raw_lgb_2y_5 = {'num_leaves': 47,
                'min_data_in_leaf': 257,
                'learning_rate': 0.011487,
                'num_boosting_rounds': 1600}

raw_lgb_2y_6 = {'num_leaves': 60,
                'min_data_in_leaf': 349,
                'learning_rate': 0.008397,
                'num_boosting_rounds': 1800}

# with fe3
# raw_lgb_2y_1 = {'num_leaves': 70,
#                 'min_data_in_leaf': 276,
#                 'learning_rate': 0.013404,
#                 'num_boosting_rounds': 950}
#
# raw_lgb_2y_2 = {'num_leaves': 49,
#                 'min_data_in_leaf': 279,
#                 'learning_rate': 0.009982,
#                 'num_boosting_rounds': 1500}
#
# raw_lgb_2y_3 = {'num_leaves': 76,
#                 'min_data_in_leaf': 256,
#                 'learning_rate': 0.014353,
#                 'num_boosting_rounds': 800}


lgb_month_1 = {'num_leaves': 5,
             'min_data_in_leaf': 74,
             'learning_rate': 0.059968,
             'num_boosting_rounds': 50}


lgb_month_2 = {'num_leaves': 6,
                'min_data_in_leaf': 61,
                'learning_rate': 0.0085312,
                'num_boosting_rounds': 1000}


lgb_month_3 = {'num_leaves': 5,
                'min_data_in_leaf': 28,
                'learning_rate': 0.029624,
                'num_boosting_rounds': 200}


lgb_month_4 = {'num_leaves': 5,
             'min_data_in_leaf': 74,
             'learning_rate': 0.059968,
             'num_boosting_rounds': 800}


lgb_month_5 = {'num_leaves': 6,
                'min_data_in_leaf': 61,
                'learning_rate': 0.0085312,
                'num_boosting_rounds': 1000}


lgb_month_6 = {'num_leaves': 5,
                'min_data_in_leaf': 28,
                'learning_rate': 0.029624,
                'num_boosting_rounds': 200}


lgb_month_sizedown_1 = {'num_leaves': 17,
             'min_data_in_leaf': 69,
             'learning_rate': 0.00717,
             'num_boosting_rounds': 200}


lgb_month_sizedown_2 = {'num_leaves': 5,
                'min_data_in_leaf': 43,
                'learning_rate': 0.001315,
                'num_boosting_rounds': 1000}


lgb_month_sizedown_3 = {'num_leaves': 17,
                'min_data_in_leaf': 38,
                'learning_rate': 0.0465,
                'num_boosting_rounds': 200}


lgb_month_sizedown_4 = {'num_leaves': 17,
             'min_data_in_leaf': 69,
             'learning_rate': 0.00717,
             'num_boosting_rounds': 800}


lgb_month_sizedown_5 = {'num_leaves': 5,
                'min_data_in_leaf': 43,
                'learning_rate': 0.001315,
                'num_boosting_rounds': 1000}


lgb_month_sizedown_6 = {'num_leaves': 17,
                'min_data_in_leaf': 38,
                'learning_rate': 0.0465,
                'num_boosting_rounds': 200}


# raw_lgb_2y = [raw_lgb_2y_1, raw_lgb_2y_2, raw_lgb_2y_3, raw_lgb_2y_4, raw_lgb_2y_5, raw_lgb_2y_6]
# lgb_month = [lgb_month_1, lgb_month_2, lgb_month_3, lgb_month_4, lgb_month_5, lgb_month_6]
# lgb_month_sizedown = [lgb_month_sizedown_1, lgb_month_sizedown_2, lgb_month_sizedown_3, lgb_month_sizedown_4, lgb_month_sizedown_5, lgb_month_sizedown_6]


raw_lgb_2y = [raw_lgb_2y_2]
lgb_month = [lgb_month_1]
lgb_month_sizedown = [lgb_month_sizedown_1]


class3_new_features = ('dollar_taxvalue_structure_land_absdiff_norm__groupby__code_zip__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_neighborhood__mean',
                       'dollar_taxvalue_structure__groupby__code_city__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__raw_census__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__str_zoning_desc__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__code_county_landuse__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__census__mean',
                       'dollar_taxvalue_structure__groupby__block__mean')


class3_rm_features = ('code_zip_lgb',
                      'code_neighborhood_lgb',
                      'code_city_lgb',
                      'raw_census_lgb',
                      'raw_block_lgb',
                      'str_zoning_desc_lgb',
                      'code_county_landuse_lgb',
                      'census_lgb',
                      'block_lgb')

step1_new_features = class3_new_features
step1_rm_features = class3_rm_features

step2_keep_only_feature = ('year_built', 'area_lot', 'dollar_tax', 'area_living_type_12', 'dollar_taxvalue_structure', 'latitude', 'longitude', 'dollar_taxvalue_land',
                           'dollar_taxvalue_structure', 'dollar_taxvalue_total', 'area_garage') + class3_new_features

step3_keep_only_feature = ('2y_diff_dollar_taxvalue_total', '2y_diff_dollar_taxvalue_land', '2y_diff_dollar_taxvalue_structure')


# lgb_year_1 = {'num_leaves': 34,
#              'min_data_in_leaf': 800,
#              'learning_rate': 0.003867,
#              'num_boosting_rounds': 900}
#
#
# sign_clf_2y_1 = {'num_leaves': 78,
#                  'min_data_in_leaf': 285,
#                  'learning_rate': 0.006248,
#                  'num_boosting_rounds': 1500}
#
# sign_clf_2y_2 = {'num_leaves': 58,
#                  'min_data_in_leaf': 233,
#                  'learning_rate': 0.006256,
#                  'num_boosting_rounds': 1500}
#
# sign_clf_2y_3 = {'num_leaves': 40,
#                  'min_data_in_leaf': 294,
#                  'learning_rate': 0.007067,
#                  'num_boosting_rounds': 1500}
#
# sign_clf_2y = [sign_clf_2y_1, sign_clf_2y_2, sign_clf_2y_3]
#
#
# sign_pos_2y_1 = {'num_leaves': 35,
#                  'min_data_in_leaf': 224,
#                  'learning_rate': 0.001427,
#                  'num_boosting_rounds': 1500}
#
# sign_pos_2y_2 = {'num_leaves': 43,
#                  'min_data_in_leaf': 307,
#                  'learning_rate': 0.004673,
#                  'num_boosting_rounds': 1500}
#
# sign_pos_2y_3 = {'num_leaves': 30,
#                  'min_data_in_leaf': 235,
#                  'learning_rate': 0.013203,
#                  'num_boosting_rounds': 1500}
#
# sign_pos_2y = [sign_pos_2y_1, sign_pos_2y_2, sign_pos_2y_3]
#
#
# sign_neg_2y_1 = {'num_leaves': 69,
#                  'min_data_in_leaf': 227,
#                  'learning_rate': 0.001398,
#                  'num_boosting_rounds': 1500}
#
# sign_neg_2y_2 = {'num_leaves': 48,
#                  'min_data_in_leaf': 353,
#                  'learning_rate': 0.007309,
#                  'num_boosting_rounds': 1500}
#
# sign_neg_2y_3 = {'num_leaves': 67,
#                  'min_data_in_leaf': 252,
#                  'learning_rate': 0.01001,
#                  'num_boosting_rounds': 1500}
#
# sign_neg_2y = [sign_neg_2y_1, sign_neg_2y_2, sign_neg_2y_3]
