#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Stress.csv")


# In[3]:


df.head()


# ## plot the percentage of stress and not stress

# In[4]:


from matplotlib import pyplot as plt


# In[5]:


bar_graph= (df["label"].value_counts(normalize=True)*100).sort_values().plot(kind='barh',figsize=(16 , 4),title='Stress and Not Stress')
for a in bar_graph.containers:
    plt.bar_label(a,fmt='%.2f%%')
plt.show()


# In[6]:


df["label"].value_counts()


# # clean the data

# In[7]:


def show(data):
    for index , content in enumerate(data) :
        
        if(index <= 5) :
            print(content)
        else :
            break


# ## 1.clean the text

# In[8]:


import re
import string


# In[9]:


def wordopt(text) :
    
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text


# In[10]:


df["text"] = df["text"].apply(wordopt)


# In[11]:


show(df["text"])


# ## 2.tokenization

# In[12]:


import nltk
nltk.download("punkt")


# In[13]:


from nltk.tokenize import word_tokenize


# In[14]:


df["text"]= df["text"].apply(word_tokenize)


# In[15]:


show(df["text"])


# ## 3.Remove Stopwards

# In[16]:


from nltk.corpus import stopwords


# In[17]:


def remove_stopwords(text) :
    stpws = set(stopwords.words("english"))
    
    filtered_text = [words for words in text if words not in stpws]
    return filtered_text


# In[18]:


df["text"]= df["text"].apply(remove_stopwords)


# In[19]:


show(df["text"])


# ## 4.Lemmatization

# In[20]:


nltk.download("wordnet")


# In[21]:


from nltk.stem import WordNetLemmatizer


# In[22]:


def lemmatization_words(text) :
    
    lemmer= WordNetLemmatizer()
    lemmatization_text= [lemmer.lemmatize(word , pos= "v") for word in text]
    
    return lemmatization_text


# In[23]:


df["text"] = df["text"].apply(lemmatization_words)


# In[24]:


show(df["text"])


# ## 5.create corpus

# In[25]:


df["text"]= df["text"].apply(lambda x : ' '.join([index for index in x]))


# In[26]:


show(df["text"])


# In[27]:


df["text"].dtype


# ## split train and test data

# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


x = df["text"]
y = df["label"]


# In[30]:


x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state= 42)


# In[31]:





# ## vectorization

# In[32]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[33]:


vetor = TfidfVectorizer()


# In[34]:


x_train = vetor.fit_transform(x_train)
x_test = vetor.transform(x_test)


# In[35]:


x_train.dtype


# ## model

# In[36]:


from sklearn.svm import SVC


# In[37]:


model = SVC()


# In[38]:


model.fit(x_train , y_train)


# ## evaluating the model

# In[39]:


from sklearn.metrics import accuracy_score , mean_squared_error


# In[40]:


y_pred = model.predict(x_test)


# In[41]:


accuracy_score(y_test , y_pred)


# In[42]:


mse = mean_squared_error(y_test , y_pred)


# In[43]:


import numpy as np


# In[44]:


rmse = np.sqrt(mse)


# In[45]:


rmse


# ## save the model

# In[46]:


from joblib import dump , load


# In[47]:


dump(model , "Stress2.joblib")


# ## test the model

# In[48]:


user = pd.DataFrame({
    "text":['i am going to commit suicide', 'love to coding' , "is AI dengerous for human"]
})


# In[49]:


user["text"] = user["text"].apply(word_tokenize)
user["text"] = user["text"].apply(remove_stopwords)
user["text"] = user["text"].apply(lemmatization_words)
user["text"] = user["text"].apply(lambda x : ' '.join([index for index in x]))


# In[50]:


user


# In[51]:


x_test = vetor.transform(user["text"])


# ## use the model

# In[52]:


my_model = load("Stress2.joblib")


# In[55]:


user["predicted"] = my_model.predict(x_test)
user["predicted"]= user["predicted"].apply(lambda x : 'Stress' if x == 1 else 'Not Stress')


# In[56]:


user


# In[ ]:




