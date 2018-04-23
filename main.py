
#description     :Two Sigma competition by Kaggle
#author          :bgw
#date            :201704
#version         :0.1
#usage           :main.py
#notes           :
#python_version  :3.6
#==============================================================================


import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import FunctionTransformer
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import re
from sklearn.feature_extraction.text import CountVectorizer
from geopy.geocoders import Nominatim

def timeparser(series):
    df_time = pd.DataFrame(index=series.index, columns=['month',
                                                         'day',
                                                         'hour',
                                                         'weekday'
                                                         ])
    series = pd.to_datetime(series, yearfirst=True)
    df_time.weekday = series.apply(lambda x: x.weekday())
    df_time.month = series.apply(lambda x: x.month)
    df_time.day = series.apply(lambda x: x.day)
    df_time.hour = series.apply(lambda x: x.hour)
    return df_time

def clean(s):
    # Remove html tags:
    cleaned = re.sub(r"(?s)<.*?>", " ", s)
    # Keep only regular chars:
    cleaned = re.sub(r"[^A-Za-z]", " ", cleaned)
    # Remove unicode chars
    cleaned = re.sub("\\\\u(.){4}", " ", cleaned)
    # Remove extra whitespace
    cleaned = re.sub(r"&nbsp;", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = cleaned.lower().split()
    cleaned = " ".join([l.strip() for l in cleaned if (len(l) > 2)])
    return cleaned

def cl_ts(series):
    series.replace(pd.np.nan, 0, inplace=True)
    series.replace(pd.np.inf, 0, inplace=True)
    return series

basePath = os.path.dirname(os.path.abspath(__file__))
df_train = pd.read_json(path_or_buf = basePath + '/train.json')
df_test = pd.read_json(path_or_buf = basePath + '/test.json')

#Number of photos and features, badrooms
df_train['num_photos'] = df_train.photos.apply(len)
df_train["num_features"] = df_train.features.apply(len)
df_train['num_photos'] = cl_ts(df_train['num_photos'])
df_train['num_features'] = cl_ts(df_train['num_features'])

df_test['num_photos'] = df_test.photos.apply(len)
df_test["num_features"] = df_test.features.apply(len)
df_test['num_photos'] = cl_ts(df_test['num_photos'])
df_test['num_features'] = cl_ts(df_test['num_features'])

df_train['priceperphoto'] = df_train.price / df_train.num_photos
df_train['photosperfeatures'] = df_train.num_photos / df_train.num_features
df_train['priceperphoto'] = cl_ts(df_train['priceperphoto'])
df_train['photosperfeatures'] = cl_ts(df_train['photosperfeatures'])

df_test['priceperphoto'] = df_test.price / df_test.num_photos
df_test['photosperfeatures'] = df_test.num_photos / df_test.num_features
df_test['priceperphoto'] = cl_ts(df_test['priceperphoto'])
df_test['photosperfeatures'] = cl_ts(df_test['photosperfeatures'])

df_train['priceperbathroom'] = df_train.price / df_train.bathrooms
df_train['priceperbedroom'] = df_train.price / df_train.bedrooms
df_train['priceperbathroom'] = cl_ts(df_train['priceperbathroom'])
df_train['priceperbedroom'] = cl_ts(df_train['priceperbedroom'])

df_test['priceperbathroom'] = df_test.price / df_test.bathrooms
df_test['priceperbedroom'] = df_test.price / df_test.bedrooms
df_test['priceperbathroom'] = cl_ts(df_test['priceperbathroom'])
df_test['priceperbedroom'] = cl_ts(df_test['priceperbedroom'])

df_test['price_lg'] = df_test.price.apply(pd.np.log)
df_train['price_lg'] = df_train.price.apply(pd.np.log)

#Time Created
time_df = timeparser(df_train.created)
time_columns = time_df.columns
df_train = pd.concat([df_train, time_df], axis=1)

time_df = timeparser(df_test.created)
time_columns = time_df.columns
df_test = pd.concat([df_test, time_df], axis=1)

#Street adress vs display adress
import re, string, pylev
regex = re.compile('[%s]' % re.escape(string.punctuation))
df_train[['street_address', 'display_address']] = df_train[['street_address', 'display_address']].applymap(lambda x: re.sub(regex, '', x))
df_train.street_address = df_train.street_address.str.lower()
df_train.display_address = df_train.display_address.str.lower()

df_test[['street_address', 'display_address']] = df_test[['street_address', 'display_address']].applymap(lambda x: re.sub(regex, '', x))
df_test.street_address = df_test.street_address.str.lower()
df_test.display_address = df_test.display_address.str.lower()


regex = re.compile('[%s]' % re.escape(string.digits))
df_train[['street_address', 'display_address']] = df_train[['street_address', 'display_address']].applymap(lambda x: re.sub(regex, '', x))
df_test[['street_address', 'display_address']] = df_test[['street_address', 'display_address']].applymap(lambda x: re.sub(regex, '', x))

distance = [pylev.levenschtein(i[0], i[1]) for i in df_train[['street_address', 'display_address']].values]
df_train = pd.concat([df_train, pd.Series(distance, index=df_train.index)], axis=1)
distance = [pylev.levenschtein(i[0], i[1]) for i in df_test[['street_address', 'display_address']].values]
df_test = pd.concat([df_test, pd.Series(distance, index=df_test.index)], axis=1)

#df_train['adress_wrong'] = pd.np.where(df_train[0] > 6, 1, 0)
#df_train['adress_right'] = pd.np.where(df_train[0] < 6, 1, 0)

df_train['west_str'] = df_train['display_address'][df_train.display_address.str.match('^west\s|^w\s')]
df_train['east_str'] = df_train['display_address'][df_train.display_address.str.match('^east\s|^e\s')]
df_train.loc[:, 'west_str'] = pd.np.where(df_train['west_str'].isnull()==False, 1, 0)
df_train.loc[:, 'east_str'] = pd.np.where(df_train['east_str'].isnull()==False, 1, 0)

df_test['west_str'] = df_test['display_address'][df_test.display_address.str.match('^west\s|^w\s')]
df_test['east_str'] = df_test['display_address'][df_test.display_address.str.match('^east\s|^e\s')]
df_test.loc[:, 'west_str'] = pd.np.where(df_test['west_str'].isnull()==False, 1, 0)
df_test.loc[:, 'east_str'] = pd.np.where(df_test['east_str'].isnull()==False, 1, 0)

#sns.factorplot(x=0, hue='interest_level', kind='count', data=df_train)




df_train['features_new'] = df_train.features.apply(lambda x: ','.join(x))
df_test['features_new'] = df_test.features.apply(lambda x: ','.join(x))

cvect_desc = CountVectorizer(stop_words='english', max_features=200)
full_sparse = cvect_desc.fit_transform(df_train.features_new)
col_desc = ['desc_' + i for i in cvect_desc.get_feature_names()]
count_vect_df_train = pd.DataFrame(full_sparse.todense(), columns=col_desc)
#binary outcome
count_vect_df_train = pd.DataFrame(pd.np.where(count_vect_df_train >= 1, 1, 0), index=df_train.index, columns=count_vect_df_train.columns)
df_train = pd.concat([df_train, count_vect_df_train], axis=1)
#feature_selection = count_vect_df.sum(axis=0)[count_vect_df.sum(axis=0).sort_values() / len(count_vect_df) > 0.05].index

full_sparse = cvect_desc.transform(df_test.features_new)
col_desc = ['desc_' + i for i in cvect_desc.get_feature_names()]
count_vect_df_test = pd.DataFrame(full_sparse.todense(), columns=col_desc)
count_vect_df_test = pd.DataFrame(pd.np.where(count_vect_df_test >= 1, 1, 0), index=df_test.index, columns=count_vect_df_test.columns)
df_test = pd.concat([df_test, count_vect_df_test], axis=1)


#Weighted attractivenes of manager
#df_new_train = pd.DataFrame(index=df_train.index, columns=['manager_level_low', 'manager_level_medium', 'manager_level_high'])
#df_new_test = pd.DataFrame(index=df_test.index, columns=['manager_level_low', 'manager_level_medium', 'manager_level_high'])
#df_train = pd.concat([df_train, df_new_train], axis=1)
#df_test = pd.concat([df_test, df_new_test], axis=1)
#for m in df_train.groupby('manager_id'):
#    test_subset = df_train[df_train.manager_id == m[0]].index
#    df_train.loc[test_subset, 'manager_level_low'] = (m[1].interest_level == 'low').mean()
#    df_train.loc[test_subset, 'manager_level_medium'] = (m[1].interest_level == 'medium').mean()
#    df_train.loc[test_subset, 'manager_level_high'] = (m[1].interest_level == 'high').mean()
#    test_subset_test = df_test[df_test.manager_id == m[0]].index
#    df_test.loc[test_subset_test, 'manager_level_low'] = (m[1].interest_level == 'low').mean()
#    df_test.loc[test_subset_test, 'manager_level_medium'] = (m[1].interest_level == 'medium').mean()
#    df_test.loc[test_subset_test, 'manager_level_high'] = (m[1].interest_level == 'high').mean()

#df_train['manager_level_low'] = cl_ts(df_train['manager_level_low'])
#df_train['manager_level_medium'] = cl_ts(df_train['manager_level_medium'])
#df_train['manager_level_high'] = cl_ts(df_train['manager_level_high'])
#df_test['manager_level_low'] = cl_ts(df_test['manager_level_low'])
#df_test['manager_level_medium'] = cl_ts(df_test['manager_level_medium'])
#df_test['manager_level_high'] = cl_ts(df_test['manager_level_high'])


###################################
# Manager ID and Buildings ID
le = LabelEncoder()
le.fit(df_train.manager_id.tolist() + df_test.manager_id.tolist())
df_train.manager_id = le.transform(df_train.manager_id.tolist())
df_test.manager_id = le.transform(df_test.manager_id.tolist())


le = LabelEncoder()
le.fit(df_test.street_address.tolist() + df_train.street_address.tolist())
df_train.street_address = le.transform(df_train.street_address.tolist())
df_test.street_address = le.transform(df_test.street_address.tolist())


le = LabelEncoder()
le.fit(df_train.building_id.tolist() + df_test.building_id.tolist())
df_train.building_id = le.transform(df_train.building_id.tolist())
df_test.building_id = le.transform(df_test.building_id.tolist())


#Testing
columns_standart = ['priceperbathroom',
                    'priceperbedroom',
                    'price',
                    'price_lg',
                    'priceperphoto',
                    'num_features',
                    'num_photos',
                    'building_id',
                    'manager_id',
                    #'manager_level_low',
                    #'manager_level_medium',
                    #'manager_level_high',
                    'latitude',
                    'longitude'

                    ]


X_test = df_test[columns_standart + time_columns.tolist() + count_vect_df_test.columns.tolist()]
labels = df_test['listing_id']
Y_train = df_train['interest_level']
X_train = df_train[columns_standart + time_columns.tolist()+ count_vect_df_train.columns.tolist()]

X_train.shape, X_test.shape, Y_train.shape

#KNN
model = LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc = round(model.score(X_train, Y_train) * 100, 2)
print (acc)

#Train XBoost
model = XGBClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
probab = model.predict_proba(X_test)
acc = round(model.score(X_train, Y_train) * 100, 2)
print (acc)
df_importance = pd.DataFrame(columns=['Factor', 'Coeff'])
df_importance['Factor'] = X_train.columns
df_importance['Coeff'] = model.feature_importances_
df_importance = df_importance.sort_values('Coeff')

thresholds = pd.unique(df_importance['Coeff'])
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)#
	#train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, Y_train)
    acc = round(selection_model.score(select_X_train, Y_train) * 100, 2)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], acc))

output = pd.DataFrame(probab, index=labels, columns=['high', 'low', 'medium'])
output = output[['high', 'medium', 'low']]
output.to_csv(basePath + '/submission.csv')

df_importance = pd.DataFrame(columns=['Factor', 'Coeff'])
df_importance['Factor'] = X_train.columns
df_importance['Coeff'] = model.feature_importances_
df_importance = df_importance.sort_values('Coeff')
print ("")

import xgboost as xgb
param = {}
param['objective'] = 'multi:softprob'
param['eta'] = 0.02
param['max_depth'] = 6
param['silent'] = 1
param['num_class'] = 3
param['eval_metric'] = "mlogloss"
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['seed'] = 321
num_rounds = 400
plst = list(param.items())
target_num_map = {'high':0, 'medium':1, 'low':2}
Y_train = pd.np.array(Y_train.apply(lambda x: target_num_map[x]))

xgtrain = xgb.DMatrix(X_train, label=Y_train)
xgtest = xgb.DMatrix(X_test)
model = xgb.train(plst, xgtrain, num_rounds)
pred_test_y = model.predict(xgtest)
output = pd.DataFrame(pred_test_y, index=labels, columns=['high', 'medium', 'low'])
output = output[['high', 'medium', 'low']]
output.to_csv(basePath + '/submission.csv')

#g = sns.PairGrid(y_vars='interest_level', x_vars=['bathrooms', 'bedrooms', 'price'], data=df)
#g.map(sns.pointplot, color=sns.xkcd_rgb["plum"])
#sns.factorplot(x="bathrooms", data=df, kind="count", hue='interest_level',
#                   palette="BuPu", size=6, aspect=1.5)


#Description
df['description_new'] = df.description.apply(lambda x: clean(x))
cvect_desc = CountVectorizer(stop_words='english', max_features=200)
full_sparse = cvect_desc.fit_transform(df.description_new)
col_desc = ['desc_' + i for i in cvect_desc.get_feature_names()]
count_vect_df = pd.DataFrame(full_sparse.todense(), columns=col_desc)

columns_desc = count_vect_df.sum(axis=0).sort_values() > 15000
columns_desc = columns_desc[columns_desc==True].index
count_vect_df = count_vect_df[columns_desc]
columns_description = pd.DataFrame(pd.np.where(count_vect_df >= 1, 1, 0), index=count_vect_df.index, columns=count_vect_df.columns)
df = pd.concat([df, columns_description], axis=1)

#Latitude/Longtitude
df.longitude = df.longitude * (-1)

# Fit model using each importance as a threshold
#thresholds = pd.unique(df_importance['Coeff'])
#for thresh in thresholds:
#    selection = SelectFromModel(model, threshold=thresh, prefit=True)
#    select_X_train = selection.transform(X_train)#
	#train model
#    selection_model = XGBClassifier()
#    selection_model.fit(select_X_train, Y_train)
#    acc = round(selection_model.score(select_X_train, Y_train) * 100, 2)
#    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], acc))
print ("")


