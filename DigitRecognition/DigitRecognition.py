#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


mnist = datasets.fetch_openml("mnist_784")


# In[3]:


mnist["data"]


# In[4]:


mnist["target"]


# In[5]:


mnist.DESCR


# In[6]:


x = mnist["data"]


# In[7]:


y = mnist["target"]


# In[8]:


x.shape


# In[9]:


y.shape


# In[10]:


import matplotlib
from matplotlib import pyplot as plt


# In[11]:


import numpy as np


# In[12]:


digit = x.to_numpy()[26001]


# In[13]:


# reashape plot it 
digit_image = digit.reshape(28 , 28)


# In[14]:


plt.imshow(digit_image , cmap = matplotlib.cm.binary)
plt.axis("off")


# In[15]:


y[26001]


# In[16]:


x_train = x[:60000]
x_test = x[60000:]


# In[17]:


y_train = y[:60000]
y_test = y[60000:]


# ## create a  detector

# In[18]:


# convert string type to integer
y_train = y_train.astype(np.int_)
y_test = y_test.astype(np.int_)


# In[19]:


y_train2 = (y_train == 7)
y_test2 = (y_test == 7)


# In[20]:


y_train2


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


# In[22]:


# classifier
# clf = LogisticRegression()
clf =  SGDClassifier()


# In[23]:


clf.fit(x_train , y_train2)


# In[24]:


clf.predict([digit])


# ## cross validation

# In[25]:


from sklearn.model_selection import cross_val_score


# In[26]:


a = cross_val_score(clf , x_train , y_train2 , cv= 3, scoring= "accuracy")


# In[27]:


a


# In[28]:


a.mean()


# In[29]:


from sklearn.model_selection import cross_val_predict


# In[30]:


y_train_pred = cross_val_predict(clf , x_train , y_train2 , cv= 3)


# In[31]:


y_train_pred


# ## confusion matrix

# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


confusion_matrix(y_train2 , y_train_pred)


# In[34]:


# when we have perfect prediction the result would be like this type
confusion_matrix(y_train2 , y_train2)


# ## precision and recall

# In[35]:


from sklearn.metrics import precision_score , recall_score


# In[36]:


precision_score(y_train2 , y_train_pred)


# In[37]:


recall_score(y_train2 ,y_train_pred)


# ## F1-Score

# In[38]:


from sklearn.metrics import f1_score


# In[39]:


f1_score(y_train2 , y_train_pred)


# ## precision recall curve

# In[40]:


from sklearn.metrics import precision_recall_curve


# In[41]:


y_score = cross_val_predict(clf , x_train , y_train2 , cv= 3 , method= "decision_function")


# In[42]:


y_score


# In[43]:


precissions , recalls , thresholds = precision_recall_curve(y_train2 , y_score)


# In[44]:


precissions


# In[45]:


recalls


# In[46]:


thresholds


# ## plot precision recall curve

# In[47]:


# precission and recall are inversely proportional to each other

plt.plot(thresholds , precissions[:-1] , "b--" , label = "precission")
plt.plot(thresholds , recalls[:-1] , "g-" , label = "recall")
plt.xlabel("threshold")
plt.legend(loc = "upper left")
plt.show()


# In[ ]:




