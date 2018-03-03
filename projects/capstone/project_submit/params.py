class3_new_features = ['dollar_taxvalue_total_dollar_tax_ratio__groupby__block__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__census__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__code_county_landuse__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__str_zoning_desc__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__raw_block__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__raw_census__mean',
                       'dollar_taxvalue_structure__groupby__code_city__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_neighborhood__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_zip__mean']

# class3_new_features = []

class3_rm_features = ['code_zip_lgb',
                      'code_neighborhood_lgb',
                      'code_city_lgb',
                      'raw_census_lgb',
                      'raw_block_lgb',
                      'str_zoning_desc_lgb',
                      'code_county_landuse_lgb',
                      'census_lgb',
                      'block_lgb']
# class3_rm_features = []

# class3_new_features = ()


step2_keep_only_feature = ['year_assess', 'year_built', 'area_lot', 'dollar_tax', 'area_living_type_12', 'dollar_taxvalue_structure', 'latitude', 'longitude', 'dollar_taxvalue_land',
                           'dollar_taxvalue_structure', 'dollar_taxvalue_total', 'area_garage'] + class3_new_features

lgb_raw = {'num_leaves': 76,
           'min_data_in_leaf': 168,
           'learning_rate': 0.0062,
           'num_boosting_rounds': 2250
           }

lgb_step1_1 = {'num_leaves': 78,
               'min_data_in_leaf': 323,
               'learning_rate': 0.00288,
               'num_boosting_rounds': 7300
               }

lgb_step1_2 = {'num_leaves': 77,
               'min_data_in_leaf': 296,
               'learning_rate': 0.002755,
               'num_boosting_rounds': 7300
               }

lgb_step1_3 = {'num_leaves': 69,
               'min_data_in_leaf': 231,
               'learning_rate': 0.0021,
               'num_boosting_rounds': 9500
               }

lgb_step1_4 = {'num_leaves': 73,
               'min_data_in_leaf': 333,
               'learning_rate': 0.002732,
               'num_boosting_rounds': 7400
               }

lgb_step1_5 = {'num_leaves': 75,
               'min_data_in_leaf': 461,
               'learning_rate': 0.002726,
               'num_boosting_rounds': 7600
               }

lgb_step1 = [lgb_step1_1, lgb_step1_2, lgb_step1_3, lgb_step1_4, lgb_step1_5]


lgb_step2_1 = {'num_leaves': 9,
               'min_data_in_leaf': 97,
               'learning_rate': 0.007783,
               'num_boosting_rounds': 240
               }

lgb_step2_2 = {'num_leaves': 9,
               'min_data_in_leaf': 97,
               'learning_rate': 0.007783,
               'num_boosting_rounds': 200
               }

lgb_step2_3 = {'num_leaves': 9,
               'min_data_in_leaf': 97,
               'learning_rate': 0.007783,
               'num_boosting_rounds': 280
               }

lgb_step2_4 = {'num_leaves': 6,
               'min_data_in_leaf': 78,
               'learning_rate': 0.005595,
               'num_boosting_rounds': 340
               }

lgb_step2_5 = {'num_leaves': 6,
               'min_data_in_leaf': 78,
               'learning_rate': 0.005595,
               'num_boosting_rounds': 280
               }

lgb_step2_6 = {'num_leaves': 6,
               'min_data_in_leaf': 78,
               'learning_rate': 0.005595,
               'num_boosting_rounds': 400
               }

lgb_step2 = [lgb_step2_1, lgb_step2_2, lgb_step2_3, lgb_step2_4, lgb_step2_5, lgb_step2_6]