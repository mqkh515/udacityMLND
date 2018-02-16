class3_new_features = ('dollar_taxvalue_structure_land_absdiff_norm__groupby__code_zip__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__code_neighborhood__mean',
                       'dollar_taxvalue_structure__groupby__code_city__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__raw_census__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__str_zoning_desc__mean',
                       'dollar_taxvalue_total_dollar_tax_ratio__groupby__code_county_landuse__mean',
                       'dollar_taxvalue_structure_land_absdiff_norm__groupby__census__mean',
                       'dollar_taxvalue_structure__groupby__block__mean')

lgb_raw = {'num_leaves': 76,
           'min_data_in_leaf': 168,
           'learning_rate': 0.0062,
           'num_boosting_rounds': 2250
           }