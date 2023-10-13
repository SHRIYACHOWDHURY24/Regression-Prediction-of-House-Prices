#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

housedata=pd.read_csv("housing.csv")


# In[87]:


housedata


# In[5]:


housedata.info()


# In[9]:


housedata.dropna(inplace=True)


# In[11]:


housedata.info()


# In[13]:


from sklearn.model_selection import train_test_split
x=housedata.drop(['median_house_value'],axis=1)
y=housedata['median_house_value']


# In[14]:


x


# In[15]:


y


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[17]:


train_data=x_train.join(y_train)


# In[18]:


train_data


# In[38]:


train_data.hist(figsize=(15,8))


# In[37]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')


# In[39]:


train_data['total_rooms']=np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']=np.log(train_data['total_bedrooms']+1)
train_data['population']=np.log(train_data['population']+1)
train_data['households']=np.log(train_data['households']+1)


# In[40]:


train_data.hist(figsize=(15,8))


# In[42]:


train_data.ocean_proximity.value_counts()


# In[43]:


pd.get_dummies(train_data.ocean_proximity)


# In[77]:


train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)


# In[46]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')


# In[48]:


plt.figure(figsize=(15,8))
sns.scatterplot(x="total_rooms",y="total_bedrooms",data=train_data,hue="median_house_value")


# In[49]:


train_data['bedroom_ratio']=train_data['total_bedrooms']/train_data['total_rooms']
train_data['household_rooms']=train_data['total_rooms']/train_data['households']


# In[50]:


plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap='YlGnBu')


# In[74]:


from sklearn.linear_model import LinearRegression
train_data=train_data.drop(['ocean_proximity'],axis=1)
x_train1=train_data.drop(['median_house_value'],axis=1)
y_train1=train_data['median_house_value']
reg=LinearRegression()
reg.fit(x_train1,y_train1)


# In[ ]:





# In[ ]:





# In[81]:


from sklearn.model_selection import train_test_split

x_train2,x_test2,y_train2,y_test2=train_test_split(x_train1,y_train1,test_size=0.2)
test_data=x_test2.join(y_test2)
                                               
test_data['total_rooms']=np.log(test_data['total_rooms']+1)
test_data['total_bedrooms']=np.log(test_data['total_bedrooms']+1)
test_data['population']=np.log(test_data['population']+1)
test_data['households']=np.log(test_data['households']+1)

test_data['bedroom_ratio']=test_data['total_bedrooms']/test_data['total_rooms']
test_data['household_rooms']=test_data['total_rooms']/test_data['households']


# In[ ]:





# In[82]:


reg.score(x_test2,y_test2)


# In[83]:


from sklearn.ensemble import RandomForestRegressor

forest=RandomForestRegressor()

forest.fit(x_train1,y_train1)


# In[88]:


forest.score(x_test2,y_test2)


# In[86]:





# In[ ]:




