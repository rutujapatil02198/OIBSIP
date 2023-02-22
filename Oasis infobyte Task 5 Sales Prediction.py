#!/usr/bin/env python
# coding: utf-8

# ## Oasis Infobyte : Data Science Internship 
# ### Task 5 Sales Prediction Using Python
# #### Name Of Intern: Rutuja Patil
# 
# ##### import Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### import Dataset

# In[2]:


df=pd.read_csv("C:/Users/rutuj/Desktop/Task5/Advertising.csv")


# In[3]:


df.head()


# In[4]:


df.size


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# ### Data Cleaning

# In[8]:


df.drop(columns=["Unnamed: 0"],axis=1 ,inplace=True)
df


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df


# #### Data Visualization

# In[12]:


sns.scatterplot(df['TV'],df['Sales'])


# In[13]:


sns.scatterplot(df['Radio'],df['Sales'])


# In[14]:


sns.scatterplot(df['Newspaper'],df['Sales'])


# #### Data Modeling

# In[15]:


x=df.drop(['Sales'],1)
df.head()


# In[16]:


y=df['Sales']
df.head()


# In[17]:


#train test split
from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[19]:


from sklearn.linear_model import LinearRegression


# In[24]:


model_lr=LinearRegression()


# In[25]:


model_lr.fit(x_train, y_train)


# In[26]:


y_pred = model_lr.predict(x_test)
y_pred


# In[27]:


coefficient=model_lr.coef_
coefficient


# In[28]:


intercept=model_lr.intercept_
intercept


# In[29]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)*100


# In[ ]:




