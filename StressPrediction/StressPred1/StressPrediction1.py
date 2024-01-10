#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Stress.csv")


# In[3]:


my_df = df[["label" , "confidence" , "social_timestamp"]]


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


from matplotlib import pyplot as plt


# In[9]:


df.hist(bins= 50 , figsize= (20 , 15))
plt.show()


# ## split test data and train data

# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[11]:


# 0 and 1 in "label" should be equally divided in train and test data

split = StratifiedShuffleSplit(n_splits= 1 , test_size= 0.2 , random_state= 42)

for train_index , test_index in split.split(my_df , my_df["label"]) :
    
    strat_train_set = my_df.loc[train_index]
    strat_test_set = my_df.loc[test_index]


# In[12]:


strat_train_set["label"].value_counts()


# In[13]:


strat_test_set["label"].value_counts()


# In[14]:


my_df = strat_train_set.copy()


# In[15]:


my_df.describe()


# ## correlation

# In[16]:


corr_matrix = my_df.corr()

corr_matrix["confidence"].sort_values(ascending= False)


# In[17]:


df[["text" , "label"]]


# In[18]:


# for model

my_df = strat_train_set.drop("confidence" , axis = 1)
my_df_labels = strat_train_set["confidence"].copy()


# ## imputer

# In[19]:


from sklearn.impute import SimpleImputer


# In[20]:


imputer = SimpleImputer(strategy= "median")
imputer.fit(my_df)


# In[21]:


imputer.statistics_.shape


# In[22]:


x = imputer.transform(my_df)


# In[23]:


my_df_tr = pd.DataFrame(x , columns= my_df.columns)


# In[24]:


my_df_tr.describe()


# ## pipeline

# In[25]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[26]:


my_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy= "median")),
    ("std_scaler" , StandardScaler())
])


# In[27]:


my_df_num_tr = my_pipeline.fit_transform(my_df)


# In[28]:


my_df_num_tr


# ## model

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


# In[30]:


import numpy as np


# In[31]:


model = RandomForestRegressor()


# In[32]:


model.fit(my_df_num_tr , my_df_labels)


# In[33]:


# take some data to analyze

some_data = my_df.iloc[:6]
some_labels = my_df_labels.iloc[:6]


# In[34]:


prepared_data = my_pipeline.transform(some_data)


# In[35]:


print(model.predict(prepared_data))


# In[36]:


print(np.array(some_labels))


# In[37]:


# take full data after analyzing

some_data = my_df
some_labels = my_df_labels


# In[38]:


prepared_data = my_pipeline.transform(some_data)


# ## evaluating the model

# In[39]:


from sklearn.metrics import mean_squared_error


# In[40]:


my_df_pred = model.predict(my_df_num_tr)


# In[41]:


mse = mean_squared_error(my_df_labels , my_df_pred)
rmse = np.sqrt(mse)


# In[42]:


print(rmse)


# ## cross validation

# In[43]:


from sklearn.model_selection import cross_val_score


# In[44]:


scores = cross_val_score(model , my_df_num_tr , my_df_labels , scoring= "neg_mean_squared_error" , cv= 10)
rmse_scores = np.sqrt(-scores)


# In[45]:


print(rmse_scores)


# In[46]:


def print_score(scores) :
    print(f"Score: {scores}")
    print(f"Mean : {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")


# In[47]:


print_score(rmse_scores)


# ## dump or save the model

# In[48]:


from joblib import dump , load


# In[49]:


dump(model , "StressConfidence.joblib")


# ## testing the model on test data

# In[50]:


x_test = strat_test_set.drop("confidence" , axis = 1)
y_test = strat_test_set["confidence"].copy()
x_test_pred = my_pipeline.transform(x_test)


# In[51]:


final_pred = model.predict(x_test_pred)
final_mse = mean_squared_error(y_test , final_pred)
final_rmse = np.sqrt(final_mse)


# In[52]:


print(final_rmse)


# In[53]:


prepared_data[6]


# ## using the model

# In[54]:


from joblib import dump , load
import numpy as np


# In[55]:


my_model = load("StressConfidence.joblib")


# In[56]:


# predict the confidence based on lavel and social_timestamp
features = np.array([[0.95266102, -1.3536602]])
my_model.predict(features)


# In[57]:


print(f"so the confidence of particular feature is: {my_model.predict(features)}")


# In[ ]:




