
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
import timeit

train = pd.read_csv('../data/sales_train_v2.csv')
test = pd.read_csv('../data/test.csv')
submission = pd.read_csv('../data/sample_submission.csv')
items = pd.read_csv('../data/items.csv')
item_cats = pd.read_csv('../data/item_categories.csv')
shops = pd.read_csv('../data/shops.csv')
print('train:', train.shape, 'test:', test.shape)

# Data present on train dataset and not in the test dataset
[c for c in train.columns if c not in test.columns]


# # All data formats

# In[2]:


train.head()


# In[3]:


items.head()


# In[4]:


item_cats.head()


# In[5]:


shops.head()


# In[6]:


test.head()


# In[7]:


submission.head()


# In[8]:


# Treat only ids as training data
items_id = items.drop(labels=['item_name'], axis = 1)
item_cats_id = item_cats.drop(labels=['item_category_name'], axis = 1)
shops_id = shops.drop(labels=['shop_name'], axis = 1)


# In[9]:


# Formatting the date feature into a monthly data feature 
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

# Grouping all daily item counts into monthly item counts
train = train.drop(['date','item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})


# In[10]:


# Monthly Mean
shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False).mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Last Month (Oct 2015)
shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})


# In[11]:


# Add Mean Feature
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])

#Add Previous Month Feature
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

#Items features
train = pd.merge(train, items_id, how='left', on='item_id')

#Item Category features
train = pd.merge(train, item_cats_id, how='left', on='item_category_id')

#Shops features
train = pd.merge(train, shops_id, how='left', on='shop_id')

train.head()


# In[12]:


# Submitting test for the following month of training data
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34

#Add Mean Feature
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)

#Add Previous Month Feature
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

#Items features
test = pd.merge(test, items_id, how='left', on='item_id')

#Item Category features
test = pd.merge(test, item_cats_id, how='left', on='item_category_id')

#Shops features
test = pd.merge(test, shops_id, how='left', on='shop_id')

test['item_cnt_month'] = 0.
test.head()


# In[13]:


col = [c for c in train.columns if c not in ['item_cnt_month']]

#Validation Hold Out Month
x_train = train[train['date_block_num']<=33]
y_train = np.log1p(x_train['item_cnt_month'].clip(0.,20.))
x_train = x_train[col]

x_test = train[train['date_block_num']==33]
y_test = np.log1p(x_test['item_cnt_month'].clip(0.,20.))
x_test = x_test[col]


# In[14]:


regressor = ensemble.RandomForestRegressor(verbose=1, n_estimators=20, n_jobs=-1, warm_start = True)
start = timeit.default_timer()

regressor.fit(x_train, y_train)

stop = timeit.default_timer()
print ("duration: " + str(stop - start))


# In[15]:


start = timeit.default_timer()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test.clip(0.,20.), regressor.predict(x_test).clip(0.,20.))))
stop = timeit.default_timer()
print ("duration: " + str(stop - start))

