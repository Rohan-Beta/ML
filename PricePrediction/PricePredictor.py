#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("PricePredection.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df["CHAS"].value_counts()


# In[7]:


df.describe()


# In[8]:


from matplotlib import pyplot as plt


# In[9]:


df.hist(bins = 50 , figsize = (20 , 15))
plt.show()


# ## split test data and train data without using sklearn

# In[10]:


import numpy as np


# In[11]:


# # for better understanding sklearn train_test_data

# def split_train_test(data , test_ratio) :
    
#     np.random.seed(42) # fix the shuffle data , wihtout this train data may read the test data
#     shuffled = np.random.permutation(len(data)) # shufffle the data
    
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[: test_set_size] # from first to test_set_size
#     train_indices = shuffled[test_set_size :] # from test_set_size to last
    
#     return data.iloc[train_indices] , data.iloc[test_indices]


# In[12]:


# # take 20% of data to test
# train_set , test_set = split_train_test(df , 0.2) # test_ratio and train_ratio is 0.2 : 0.8


# In[13]:


# print(f"train set rows: {len(train_set)}\ntest set rows: {len(test_set)}")


# ## using sklearn split test data and train data

# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


train_set , test_set = train_test_split(df , test_size= 0.2 , random_state= 42)


# In[16]:


print(f"train set rows: {len(train_set)}\ntest set rows: {len(test_set)}")


# In[17]:


from sklearn.model_selection import StratifiedShuffleSplit


# In[18]:


# 0 and 1 in "CHAS" should be equally divided in train and test data

split = StratifiedShuffleSplit(n_splits = 1, test_size= 0.2 , random_state = 42)

for train_index , test_index in split.split(df , df["CHAS"]) :
    
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[19]:


strat_test_set


# In[20]:


strat_test_set["CHAS"].value_counts()


# In[21]:


strat_train_set["CHAS"].value_counts()


# In[22]:


# create a copy of train data
df = strat_train_set.copy()


# In[23]:


df.describe()


# ## correlation

# In[24]:


corr_matrix = df.corr()


# In[25]:


# example if we increase the RM value the price will also increase , strong positive correlation
# example if we increase the LSTAT value the price will decrease , strong negative correlation

corr_matrix["MEDV"].sort_values(ascending= False)


# In[26]:


from pandas.plotting import scatter_matrix


# In[27]:


attributes = ["MEDV" , "RM" , "ZN" , "LSTAT"]

scatter_matrix(df[attributes] , figsize= (12 , 8) , alpha = 0.5)


# In[28]:


# it shows that 5 room price and 9 room price same
plt.scatter(df["RM"] , df["MEDV"] , alpha = 0.8)
plt.show()


# ## attribute combinations

# In[29]:


# create a new attribute
df["TAXRM"] = df["TAX"] / df["RM"]


# In[30]:


df.head()


# In[31]:


corr_matrix = df.corr()
corr_matrix["MEDV"].sort_values(ascending= False)


# In[32]:


plt.scatter(df["TAXRM"] , df["MEDV"] , alpha = 0.8)
plt.show()


# In[33]:


# for model

df = strat_train_set.drop("MEDV" , axis= 1)
df_labels = strat_train_set["MEDV"].copy() # MEDV is the label


# ## If there is any missing Attributes

# ## imputer

# In[34]:


from sklearn.impute import SimpleImputer


# In[35]:


# we use imputer to fill the missing value in a dataset
imputer = SimpleImputer(strategy= "median")
imputer.fit(df)


# In[36]:


imputer.statistics_.shape


# In[37]:


x = imputer.transform(df)


# In[38]:


df_tr = pd.DataFrame(x , columns= df.columns)


# In[39]:


df_tr.describe()


# ## pipeline

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[41]:


my_pipeline = Pipeline([
    ("imputer" , SimpleImputer(strategy= "median")),
    ("std_scaler" , StandardScaler())
])


# In[42]:


df_num_tr = my_pipeline.fit_transform(df)


# In[43]:


df_num_tr


# ## model

# In[44]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[45]:


# model = LinearRegression() # by using linear regression we get more rmse error
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(df_num_tr , df_labels)


# In[46]:


# take some data and label to check the prediction

some_data = df.iloc[:6]
some_labels = df_labels.iloc[:6]


# In[47]:


prepared_data = my_pipeline.transform(some_data)


# In[48]:


print(model.predict(prepared_data))


# In[49]:


print(np.array(some_labels)) # compare the prediction to original


# ## evaluating the model

# In[50]:


# mean square error
from sklearn.metrics import mean_squared_error


# In[51]:


df_pred = model.predict(df_num_tr)


# In[52]:


mse = mean_squared_error(df_labels , df_pred)
rmse = np.sqrt(mse)


# In[53]:


print(rmse)


# ## cross validation

# In[54]:


from sklearn.model_selection import cross_val_score


# In[55]:


scores = cross_val_score(model , df_num_tr , df_labels , scoring= "neg_mean_squared_error" , cv= 10)
rmse_score = np.sqrt(-scores)


# In[56]:


print(rmse_score)


# In[57]:


def print_score(scores) :
    print(f"Score: {scores}")
    print(f"Mean: {scores.mean()}")
    print(f"Standard Deviation: {scores.std()}")


# In[58]:


print_score(rmse_score)


# ## dump or save the model

# In[59]:


from joblib import dump , load


# In[60]:


dump(model , "PricePrediction.joblib")


# ## Testing the model on test data

# In[61]:


x_test = strat_test_set.drop("MEDV" , axis= 1)
y_test = strat_test_set["MEDV"].copy()
x_test_pred = my_pipeline.transform(x_test)


# In[62]:


final_pred = model.predict(x_test_pred)
final_mse = mean_squared_error(y_test , final_pred)
final_rmse = np.sqrt(final_mse)


# In[63]:


print(final_rmse)


# In[64]:


# for PriceModel Testing
# predicted price compare to index number of MEDV column

prepared_data[0]


# ## Using the model

# In[65]:


from joblib import dump , load
import numpy as np


# In[66]:


# load the model
my_model = load("PricePrediction.joblib")


# In[67]:


# features of prepared_data[0] , it gives the predicted price
# compare to MEDV column of respective index

features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])

my_model.predict(features)


# In[ ]:




