
# categorical var analysis block

print('data type:' + str(prop_data['airconditioningtypeid'].dtype))
cat_num_to_str('airconditioningtypeid')
cat_feature_analysis('airconditioningtypeid')
prop_missing, train_missing = missing_ratio('airconditioningtypeid')
if (prop_missing + train_missing) < 1.5:
    visual_analysis_cat('airconditioningtypeid', 'type_air_conditioning')


# numerical var analysis block

prop_missing, train_missing = missing_ratio('basementsqft')
if prop_missing + train_missing < 1.5:
    visual_analysis_num('basementsqft', 'area_base_finished')


# add feature
feature_info['airconditioningtypeid'] = ['type_air_conditioning', 'cat', 2, prop_missing, train_missing]
categorical_groups_keep_list['airconditioningtypeid'] = ['1', '13']

# pairs
num_bathroom_assessor, num_bathroom_zillow
area_living_finished_calc, area_living_type_12
3 types of dollar values: {dollar_taxvalue_structure, dollar_taxvalue_total, dollar_taxvalue_land}

# direct impute missing
fireplacecnt
fullbathcnt

# potentially can be improved
num_fireplace: ({1, nan}, other)
num_fullbath: (1, 2, other)
area_garage: use group mean to impute
unitcnt: group 2-4, mark higher as NaN
numberofstories: gorup 2,3, mark 4 as nan, make categorical

# row removal


