# class3_new_features = ('dollar_taxvalue_structure_land_absdiff_norm__groupby__code_zip__mean',
#                        'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_neighborhood__mean',
#                        'dollar_taxvalue_structure__groupby__code_city__mean',
#                        'dollar_taxvalue_structure_land_absdiff_norm__groupby__raw_census__mean',
#                        'dollar_taxvalue_structure_land_absdiff_norm__groupby__str_zoning_desc__mean',
#                        'dollar_taxvalue_total_dollar_tax_ratio__groupby__code_county_landuse__mean',
#                        'dollar_taxvalue_structure_land_absdiff_norm__groupby__census__mean',
#                        'dollar_taxvalue_structure__groupby__block__mean')


class3_new_features = ['dollar_taxvalue_total_dollar_tax_ratio__groupby__block__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__census__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__code_county_landuse__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__str_zoning_desc__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__raw_block__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__raw_census__mean',
                       'dollar_taxvalue_structure__groupby__code_city__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_neighborhood__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_zip__mean']


class3_rm_features = ['code_zip_lgb',
                      'code_neighborhood_lgb',
                      'code_city_lgb',
                      'raw_census_lgb',
                      'raw_block_lgb',
                      'str_zoning_desc_lgb',
                      'code_county_landuse_lgb',
                      'census_lgb',
                      'block_lgb']


# class3_new_features = ()


step2_keep_only_feature = ['year_assess', 'year_built', 'area_lot', 'dollar_tax', 'area_living_type_12', 'dollar_taxvalue_structure', 'latitude', 'longitude', 'dollar_taxvalue_land',
                           'dollar_taxvalue_structure', 'dollar_taxvalue_total', 'area_garage'] + class3_new_features

lgb_raw = {'num_leaves': 76,
           'min_data_in_leaf': 168,
           'learning_rate': 0.0062,
           'num_boosting_rounds': 2250
           }