# raw_lgb_2y_1 = {'num_leaves': 53,
#                 'min_data_in_leaf': 366,
#                 'learning_rate': 0.002382,
#                 'num_boosting_rounds': 10500}
#
# raw_lgb_2y_2 = {'num_leaves': 64,
#                 'min_data_in_leaf': 467,
#                 'learning_rate': 0.002088,
#                 'num_boosting_rounds': 11000}
#
# raw_lgb_2y_3 = {'num_leaves': 61,
#                 'min_data_in_leaf': 344,
#                 'learning_rate': 0.003809,
#                 'num_boosting_rounds': 6655}
#
# raw_lgb_2y_4 = {'num_leaves': 45,
#                 'min_data_in_leaf': 297,
#                 'learning_rate': 0.002589,
#                 'num_boosting_rounds': 11500}
#
# raw_lgb_2y_5 = {'num_leaves': 73,
#                 'min_data_in_leaf': 264,
#                 'learning_rate': 0.0082,
#                 'num_boosting_rounds': 2400}


raw_lgb_2y_1 = {'num_leaves': 53,
                'min_data_in_leaf': 366,
                'learning_rate': 0.002382,
                'num_boosting_rounds': 9400}

raw_lgb_2y_2 = {'num_leaves': 64,
                'min_data_in_leaf': 467,
                'learning_rate': 0.002088,
                'num_boosting_rounds': 9600}

raw_lgb_2y_3 = {'num_leaves': 61,
                'min_data_in_leaf': 344,
                'learning_rate': 0.003809,
                'num_boosting_rounds': 5800}

raw_lgb_2y_4 = {'num_leaves': 45,
                'min_data_in_leaf': 297,
                'learning_rate': 0.002589,
                'num_boosting_rounds': 10000}

raw_lgb_2y_5 = {'num_leaves': 73,
                'min_data_in_leaf': 264,
                'learning_rate': 0.0082,
                'num_boosting_rounds': 2300}



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


# lgb_month_1 = {'num_leaves': 7,
#              'min_data_in_leaf': 37,
#              'learning_rate': 0.004607,
#              'num_boosting_rounds': 360}
#
#
# lgb_month_2 = {'num_leaves': 7,
#              'min_data_in_leaf': 37,
#              'learning_rate': 0.004607,
#              'num_boosting_rounds': 480}
#
#
# lgb_month_3 = {'num_leaves': 6,
#                 'min_data_in_leaf': 82,
#                 'learning_rate': 0.003154,
#                 'num_boosting_rounds': 600}
#
#
# lgb_month_4 = {'num_leaves': 6,
#                 'min_data_in_leaf': 82,
#                 'learning_rate': 0.003154,
#                 'num_boosting_rounds': 800}
#
#
# lgb_month_sizedown_1 = {'num_leaves': 10,
#              'min_data_in_leaf': 77,
#              'learning_rate': 0.002,
#              'num_boosting_rounds': 620}
#
#
# lgb_month_sizedown_2 = {'num_leaves': 10,
#              'min_data_in_leaf': 77,
#              'learning_rate': 0.002,
#              'num_boosting_rounds': 860}
#
#
# lgb_month_sizedown_3 = {'num_leaves': 12,
#                 'min_data_in_leaf': 72,
#                 'learning_rate': 0.001035,
#                 'num_boosting_rounds': 1000}
#
#
# lgb_month_sizedown_4 = {'num_leaves': 12,
#                 'min_data_in_leaf': 72,
#                 'learning_rate': 0.001035,
#                 'num_boosting_rounds': 1450}


lgb_month_1 = {'num_leaves': 7,
             'min_data_in_leaf': 37,
             'learning_rate': 0.004607,
             'num_boosting_rounds': 360}


lgb_month_2 = {'num_leaves': 7,
             'min_data_in_leaf': 37,
             'learning_rate': 0.004607,
             'num_boosting_rounds': 480}


lgb_month_3 = {'num_leaves': 7,
             'min_data_in_leaf': 37,
             'learning_rate': 0.004607,
             'num_boosting_rounds': 320}

p.lgb_month_mon10 = []


lgb_month_4 = {'num_leaves': 7,
             'min_data_in_leaf': 37,
             'learning_rate': 0.004607,
             'num_boosting_rounds': 520}


# lgb_month_sizedown_1 = {'num_leaves': 10,
#              'min_data_in_leaf': 77,
#              'learning_rate': 0.002,
#              'num_boosting_rounds': 620}
#
#
# lgb_month_sizedown_2 = {'num_leaves': 10,
#              'min_data_in_leaf': 77,
#              'learning_rate': 0.002,
#              'num_boosting_rounds': 860}
#
#
# lgb_month_sizedown_3 = {'num_leaves': 12,
#                 'min_data_in_leaf': 72,
#                 'learning_rate': 0.001035,
#                 'num_boosting_rounds': 1000}
#
#
# lgb_month_sizedown_4 = {'num_leaves': 12,
#                 'min_data_in_leaf': 72,
#                 'learning_rate': 0.001035,
#                 'num_boosting_rounds': 1450}


raw_lgb_2y = [raw_lgb_2y_1, raw_lgb_2y_2, raw_lgb_2y_3, raw_lgb_2y_4, raw_lgb_2y_5]
lgb_month = [lgb_month_1, lgb_month_2, lgb_month_3, lgb_month_4]
# lgb_month_sizedown = [lgb_month_sizedown_1, lgb_month_sizedown_2, lgb_month_sizedown_3, lgb_month_sizedown_4]


# raw_lgb_2y = [raw_lgb_2y_2]
# lgb_month = [lgb_month_1]
# lgb_month_sizedown = [lgb_month_sizedown_1]


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
