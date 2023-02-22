#!/usr/bin/env python
# coding: utf-8

# ### Unemployement Analysis With Python 
# #### OASIS INFOBYTE : DATA SCIENCE INTERN
# #### BATCH : FEB 2023
# #### Inten : Rutuja Patil
# 
# 
# 
# 
# #### Import Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import datetime as dt
import calendar
import plotly.graph_objects as go


# #### Import dataset

# In[2]:


df=pd.read_csv("C:/Users/rutuj/Desktop/Umemployment analysis/Unemployment_Rate_upto_11_2020.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


## Checking the missing values 
df.isnull().sum()


# In[7]:


## Checking correlation in features in datasets
df.corr()


# In[8]:


## Correlation matrics
sns.heatmap(df.corr(),annot=True)


# In[9]:


## Unemployement rate accourding to different regions of india
df.columns=["States","Date","Freqeuncy",
           "Estimated Unemployment Rate","Estimatted Employed","Estmated Labour Participation Rate","Region",
            "longitude","latitide"]
plt.figure(figsize=(10,8))
plt.title("indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue='Region',data=df)
plt.show()


# In[10]:


sns.pairplot(df)


# In[11]:


unemployment = df[["States","Region","Estimated Unemployment Rate"]]
figure=px.sunburst(unemployment,path=['Region','States'],
                  values="Estimated Unemployment Rate",
                  width=500,height=500,color_continuous_scale="RdYlGn",
                  title="Unemployment Rate in India")


# In[ ]:




