#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("mail_data.csv")


# In[3]:


df.head()


# In[4]:


df["Category"].value_counts()


# In[5]:


df.isnull().sum()


# In[6]:


df.loc[df["Category"] == "ham" , "Label",] = 0
df.loc[df["Category"] == "spam" , "Label",] = 1


# In[7]:


df.head()


# In[8]:


x = df["Message"]
y = df["Label"].convert_dtypes(int)


# In[9]:


print(y)


# In[10]:


df["Message"][0]


# In[11]:


df["Message"] = df["Message"].apply(lambda x : x.lower())


# In[12]:


df["Message"][0]


# ## split train and test data

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state= 42)


# ## vectorization

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[16]:


vector = TfidfVectorizer()


# In[17]:


x_train = vector.fit_transform(x_train)
x_test = vector.transform(x_test)


# ## model

# In[18]:


from sklearn.linear_model import LogisticRegression


# In[19]:


model = LogisticRegression()


# In[20]:


model.fit(x_train , y_train)


# ## evaluating the model

# In[21]:


from sklearn.metrics import accuracy_score , mean_squared_error
import numpy as np


# In[22]:


# accuracy on test data
y_pred = model.predict(x_test)


# In[23]:


accuracy_score(y_test , y_pred)


# In[24]:


mse = mean_squared_error(y_test , y_pred)


# In[25]:


rmse = np.sqrt(mse)
rmse


# ## save the model

# In[26]:


from joblib import dump , load


# In[27]:


dump(model , "EMail_Spam_Detection.joblib")


# ## test the model

# In[28]:


user = pd.DataFrame({
    "text" : [input("Your email: ")]
})


# In[29]:


user


# In[30]:


user_data_feature = vector.transform(user["text"])


# ## use the model

# In[31]:


my_model = load("EMail_Spam_Detection.joblib")


# In[32]:


user["predicted"] = my_model.predict(user_data_feature)
user["predicted"] = user["predicted"].apply(lambda x : "Spam" if x == 1 else "Not Spam")


# In[33]:


user


# In[ ]:




