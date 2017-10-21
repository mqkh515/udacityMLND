import pickle as pkl

prop_2016 = pkl.load(open('prop_2016_cache.pkl', 'rb'))
prop_2017 = pkl.load(open('prop_2017_cache.pkl', 'rb'))

train_x = pkl.load(open('train_x_all.pkl', 'rb'))
train_y = pkl.load(open('train_y_all.pkl', 'rb'))