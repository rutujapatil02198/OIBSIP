#!/usr/bin/env python
# coding: utf-8

# ### IRIS FLOWER CLASSIFICATION ML PROJECT 
# 
# ### OASIS INFOBYTE : DATA SCIENCE INTERN
# ####  BATCH : FEB 2023
# #### AUTHOR : RUTUJA PATIL

# #### Import Libraries  

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


# ##### Importing dataset  

# In[2]:


iris=pd.read_csv("C:/Users/rutuj/Desktop/Task1/Iris.csv")


# In[3]:


iris.head()


# In[4]:


iris.shape


# In[5]:


iris.size


# In[6]:


iris.info()


# In[7]:


iris.Species.value_counts()


# In[8]:


iris.describe()


# ##### Data Cleaning

# In[9]:


## Drop id column 
iris=iris.drop('Id',axis=1)
iris


# In[10]:


iris.isnull().sum()


# #### Exploratory Data Analysis 

# In[11]:


# Outlier detection 
plt.figure(figsize=(16,4))
plt.subplot(1,4,1)
sns.boxplot(data=iris, y="SepalLengthCm")
plt.subplot(1,4,2)
sns.boxplot(data=iris, y="SepalWidthCm",color="red")
plt.subplot(1,4,3)
sns.boxplot(data=iris,y="PetalLengthCm",color="yellow")
plt.subplot(1,4,4)
sns.boxplot(data=iris,y="PetalWidthCm",color="green")
plt.show()


# In[12]:


sns.pairplot(iris, hue="Species",palette='hls')


# In[13]:


fig, (ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x="SepalLengthCm",y="PetalLengthCm" ,hue='Species',data= iris,ax=ax1)
sns.scatterplot(x="SepalWidthCm",y="PetalWidthCm" ,hue='Species',data= iris,ax=ax2)


# ##### Correlation matrix

# In[14]:


sns.heatmap(iris.corr(),annot=True)


# ##### from boxplot it  is clear that sepal width has some outlier
# ##### from Scatter plot it is clear that iris-setosa class is away from iris-versicolor class and iris-virginics class 
# ##### from correlation matrics it isclear that Petal Length and Petal Width is highly correlated
# 

# ## Lable Encoding

# In[15]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[16]:


iris["Species"]=le.fit_transform(iris['Species'])
iris.head()


# ###### Model Tranning

# In[17]:


x=iris.drop(['Species'],1)
x


# In[18]:


y=iris['Species']
y.head()


# In[19]:


#train_test split
from sklearn.model_selection import train_test_split


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# ##### Logistic Regression model

# In[21]:


from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression()


# In[22]:


model_lr.fit(x_train,y_train)


# In[23]:


print("Accuracy of Logistic Regression Model :",model_lr.score(x_test,y_test)*100)


# ##### Decision Tree Classifier

# In[24]:


from sklearn.tree import DecisionTreeClassifier
model_dtc=DecisionTreeClassifier()


# In[25]:


model_dtc.fit(x_train,y_train)


# In[26]:


print("Accuracy of Decision Tree Classifier Model :",model_dtc.score(x_test,y_test)*100)


# ##### K-Nearest Neighbors

# In[27]:


from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier()


# In[28]:


model_knn.fit(x_train,y_train)


# In[29]:


print("Accuracy of KNN K-Neighbors Model :",model_knn.score(x_test,y_test)*100)


# #####  Thank You
