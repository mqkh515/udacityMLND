def multi_trade_analysis(error_data_inp):
    """an asset can be traded more than once within a year, can multi_trade related to large error? should we exclude these samples from training?"""
    error_data = error_data_inp.copy()
    print('average abs logerror in all traning transactions: %4.4f' % np.abs(error_data['logerror']).mean())  # 0.0684
    n_trade = error_data['logerror'].groupby(error_data['parcelid']).count()
    multi_trade_count = n_trade[n_trade > 1]
    print('number of parcels traded more than once in training data: %d' % len(multi_trade_count))  # 124, number too small
    multi_trade_parcel_set = set(multi_trade_count.index.values)
    error_data['is_multi_trade'] = error_data['parcelid'].apply(lambda x: x in multi_trade_parcel_set)
    error_data_multi_trade = error_data[error_data['is_multi_trade']]
    print('average abs logerror in all multi_trade transactions: %4.4f' % np.abs(error_data


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


