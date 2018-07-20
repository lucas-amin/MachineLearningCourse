
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
from matplotlib import pylab as plt
import timeit

# Loading all data files
directory = '../data/'
train = pd.read_csv(directory + 'sales_train_v2.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
test = pd.read_csv(directory + 'test.csv')
submission = pd.read_csv(directory + 'sample_submission.csv')
items = pd.read_csv(directory + 'items.csv')
item_cats = pd.read_csv(directory + 'item_categories.csv')
shops = pd.read_csv(directory + 'shops.csv')


# In[2]:


# Removing unnecesary features for this model 
train_clean = train.drop(labels = ['date', 'item_price'], axis = 1)

# Change the item count per day to item count per month by using grouping
train_clean = train_clean.groupby(["item_id","shop_id","date_block_num"]).sum().reset_index()
train_clean = train_clean.rename(index=str, columns = {"item_cnt_day":"item_cnt_month"})


# In[3]:


# Data preprocessing cell
num_month = train['date_block_num'].max()
month_list = [i for i in range(num_month + 1)] 

# Shop and item selection
shops = [54] * (num_month + 1) 
items = [22167] * (num_month + 1)

check = train_clean[["shop_id","item_id","date_block_num","item_cnt_month"]]
check = check.loc[check['shop_id'] == 54]
check = check.loc[check['item_id'] == 22167]

months_full = pd.DataFrame({'shop_id': shops, 'item_id': items, 'date_block_num':month_list})

sales_33month = pd.merge(check, months_full, how='right', on=['shop_id','item_id','date_block_num'])
sales_33month = sales_33month.sort_values(by=['date_block_num'])
sales_33month.fillna(0.00, inplace=True)

df = sales_33month[['shop_id','item_id','date_block_num','item_cnt_month']].reset_index()
df = df.drop(labels = ['index'], axis = 1)

x, y = df, df.item_cnt_month


# In[4]:


from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

x_train = x_train.drop(["item_cnt_month"], axis=1)
x_test = x_test.drop(["item_cnt_month"], axis=1)


# In[5]:


x_train


# In[6]:


# Reshape the data between -1 and 1 and to 3D
from sklearn.preprocessing import RobustScaler, MinMaxScaler
scaler = RobustScaler()
x_train_scaled = scaler.fit_transform(x_train)

x_train_reshaped = x_train_scaled.reshape((x_train_scaled.shape[0], 1, x_train_scaled.shape[1]))
y_train_reshaped = y_train.as_matrix().reshape(y_train.shape[0], )

x_test_scaled = scaler.fit_transform(x_test)
x_test_reshaped = x_test_scaled.reshape((x_test_scaled.shape[0], 1, x_test_scaled.shape[1]))


# In[7]:


# Model instantiation

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.core import Dropout, Activation

model = Sequential()

model.add(LSTM(15, input_shape=(1, 3), return_sequences=True, activation='tanh'))
model.add(Dense(1))

model.add(Dropout(0.1))
  
model.add(LSTM(33)) 
model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['mean_squared_error', 'accuracy'])

model.summary()


# In[8]:


start = timeit.default_timer()
history = model.fit(x_train_reshaped, y_train_reshaped,  epochs = 100, batch_size = 33, verbose=1, shuffle = False, validation_split = 0.95)
stop = timeit.default_timer()

# Duration
print (stop - start)


# In[9]:


start = timeit.default_timer()
y_predicted = model.predict(x_test_reshaped)
stop = timeit.default_timer()

print (stop - start)


# In[10]:


from sklearn.metrics import mean_squared_error
from numpy import sqrt
rmse = sqrt(mean_squared_error(y_test, y_predicted))
print('Val RMSE: %.3f' % rmse)

