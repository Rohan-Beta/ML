#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


arr = np.arange(1000000)


# In[4]:


pylist = list(range(1000000))


# In[6]:


get_ipython().run_line_magic('time', 'for item in range(10): [item * 3 for item in pylist]')


# In[10]:


get_ipython().run_line_magic('time', 'for item in range(10): arr = arr*3')


# In[11]:


print(type(arr))


# In[ ]:




