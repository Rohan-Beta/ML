#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("movies.csv")


# In[3]:


df.head()


# In[4]:


df["genres"][0]


# In[5]:


df["keywords"][0]


# ## clean text in genres and keywords

# In[6]:


import ast


# In[7]:


def convert(text) :
    l = []
    
    for i in ast.literal_eval(text) :
        l.append(i["name"])
    
    return l


# In[8]:


df["genres"] = df["genres"].apply(convert)
df["keywords"] = df["keywords"].apply(convert)


# In[9]:


df["genres"][0]


# In[10]:


df["keywords"][0]


# ## split text in overview column

# In[11]:


df["overview"][0]


# In[12]:


df["overview"].isnull().sum()


# In[13]:


df["overview"] = df["overview"].fillna("")


# In[14]:


df["overview"].isnull().sum()


# In[15]:


df["overview"] = df["overview"].apply(lambda x: x.split())


# In[16]:


df["overview"][0] # text space is already handle in this column


# ## handle the text space

# In[17]:


df["genres"][0]


# In[18]:


df["keywords"][0]


# In[19]:


df["genres"] = df["genres"].apply(lambda x : [i.replace(" " , "") for i in x])
df["keywords"] = df["keywords"].apply(lambda x : [i.replace(" " , "") for i in x])


# In[20]:


df["genres"][0]


# In[21]:


df["keywords"][0]


# ## create a column that stores all merged info

# In[22]:


df["info"] = df["overview"] + df["genres"] + df["keywords"]


# In[23]:


df.head(2)


# In[24]:


df["info"][0]


# In[25]:


df["info"] = df["info"].apply(lambda x : ' '.join(x))


# In[26]:


df["info"][0]


# In[27]:


df["info"] = df["info"].apply(lambda x : x.lower())


# In[28]:


df["info"][0]


# ## vectorization

# In[29]:


from sklearn.feature_extraction.text import CountVectorizer


# In[30]:


cv = CountVectorizer()


# In[31]:


vector = cv.fit_transform(df["info"]).toarray()


# In[32]:


vector[0]


# ## porter stemmer

# In[33]:


from nltk.stem.porter import PorterStemmer


# In[34]:


ps = PorterStemmer()


# In[35]:


def stem(text) :
    y = []
    
    for i in text.split() :
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[36]:


df["info"] = df["info"].apply(stem)


# In[37]:


df["info"][0]


# ## cosine similarity

# In[38]:


# cosine similarity measures the simalarity between two vectors

from sklearn.metrics.pairwise import cosine_similarity


# In[39]:


cosine_similarity(vector)


# In[40]:


cs = cosine_similarity(vector)


# In[41]:


cs[0]


# ## recommendation

# In[42]:


def recommendation(movie) :
    movie_index = df[df["title"] == movie].index[0]
    
    distance = cs[movie_index]
    movie_list = sorted(list(enumerate(distance)) , reverse= True , key= (lambda x : x[1]))[1:6]
    
    for i in movie_list :
        print(df["title"][i[0]])


# In[43]:


# recommendation("Avatar")


# In[56]:


user = input("Enter a movie name: ")


# In[57]:


print("The recommendations are:\n")
recommendation(user)


# In[ ]:




