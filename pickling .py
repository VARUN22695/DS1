#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("airfoil_self_noise.dat",sep="\t")


# In[3]:


df


# In[4]:


df.header=None


# In[5]:


df


# In[6]:


df=pd.read_csv("airfoil_self_noise.dat",sep="\t",header=None)


# In[7]:


df


# In[8]:


df.columns=["Freq","Angle","Chord Length","FS vel","suction","pressure level"]


# In[9]:


df


# In[10]:


df


# In[11]:


df["Freq"].isnull()


# In[26]:


df["Freq"].isnull().sum()


# In[12]:


df


# In[13]:


x=df.iloc[:,:-1]


# In[14]:


x


# In[15]:


y=df.iloc[:,-1]


# In[16]:


y


# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.33, random_state=42)


# In[25]:


from sklearn.ensemble import RandomForestRegressor


# In[26]:


from sklearn.metrics import accuracy_score


# In[27]:


rf_model = RandomForestRegressor(n_estimators=100,random_state=42)


# In[28]:


rf_model.fit(x_train,y_train)


# In[29]:


y_pred= rf_model.predict(x_test)


# In[31]:


from sklearn.metrics import mean_absolute_error


# In[32]:


acc=mean_absolute_error(y_pred,y_test)


# In[33]:


acc


# In[55]:


from sklearn.metrics import r2_score


# In[56]:


acc1= r2_score(y_test,y_pred)


# In[57]:


acc1


# In[58]:


import numpy as np


# In[59]:


rmse= np.sqrt(acc)


# In[60]:


rmse


# In[32]:


from sklearn.model_selection import train_test_split


# In[34]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[35]:


from sklearn.linear_model import LinearRegression


# In[36]:


regression= LinearRegression()


# In[37]:


regression.fit(x_train,y_train)


# In[38]:


y_test_pred= regression.predict(x_test)


# In[39]:


y_train_pred= regression.predict(x_test)


# In[40]:


from sklearn.metrics import mean_squared_error


# In[41]:


mean_squared_error(y_train_pred,y_test_pred)


# In[17]:


df


# In[18]:


x


# In[19]:


y


# In[20]:


## how to load the pickle


# In[43]:


import pickle


# In[44]:


pickle.dump(regression,open("model.pkl","wb")) ##model.pickle is the name  of the file


# In[56]:


##how_to_dump_the data


# In[57]:


pickled_model=pickle.load(open("model.pkl",'rb'))
                               


# In[58]:


pickled_model.predict(x_test)


# In[59]:


x_test


# In[60]:


pickled_model.predict([[400,0.0,0.3048,31.7,0.003313]])


# In[61]:


df


# In[62]:


df.iloc[52,:]


# In[63]:


df.iloc[51,:]


# In[ ]:




