#Oleksandr Levchenko
#Funnel Analysis

import pandas as pd
import os
import seaborn as sns

#Import input files
basePath = os.path.dirname(os.path.abspath(__file__))
df_users = pd.read_csv('user_table.csv', parse_dates=['date'])
df_hp = pd.read_csv('home_page_table.csv')
df_st = pd.read_csv('search_page_table.csv')
df_pt = pd.read_csv('payment_page_table.csv')
df_conf = pd.read_csv('payment_confirmation_table.csv')

#Combine tables
df_comb = pd.merge(df_users, df_hp, how='left', on='user_id')
df_comb = df_comb.merge(df_st, how='left', on='user_id', suffixes=['_a', '_b'])
df_comb = df_comb.merge(df_pt, how='left', on='user_id')
df_comb = df_comb.merge(df_conf, how='left', on='user_id', suffixes=['_c', '_d'])
df_comb.sort('date')
df_comb.rename(columns={'page_a': 'home', 'page_b':'search', 'page_c':'payment', 'page_d':'confirmation'}, inplace=True)

#EDA
#Statistic about users
df_comb.date.describe()
df_comb.groupby(['sex']).count()['user_id'] / len(df_comb) #About 50 % of men and 50% of women
df_comb.groupby(['device']).count()['user_id'] / len(df_comb)  #About 66% using Desktop, 33% using Mobile

sns.plt.title('# of users per device and per sex')
sns.countplot(hue="device", x='sex', data=df_comb) # Mens and womans using D and M equaly
#Who visited search site

df_comb.groupby('sex')['home', 'search', 'payment', 'confirmation'].count()

#Time analysis
df_comb['month'] = df_comb.date.apply(lambda x: x.month)
sns.countplot(hue="device", x='month', data=df_comb)


