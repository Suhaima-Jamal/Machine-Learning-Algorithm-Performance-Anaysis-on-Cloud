#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[ ]:


import numpy as np


# In[4]:


bucket= 'bucket-dataset/Regression'
data_key= 'IRIS.csv'
data_location= 's3://{}/{}'.format(bucket,data_key) 
df=pd.read_csv(data_location)


# In[5]:


df


# In[ ]:


x = x.astype('float32')
y = y.astype('float32')


# In[6]:


df.head()


# In[4]:


from sklearn.model_selection import train_test_split 
training_data = df.sample(frac=0.7, random_state=25) 
testing_data= df.drop(training_data.index)
print(f"No. of training examples: {training_data.shape[0]}") 
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[ ]:


x_train.shape


# In[ ]:


x_test.shape


# In[ ]:


x_train=x_train.astype('float32')


# In[ ]:


y_train=y_train.astype('float32')


# In[5]:


import boto3
import sagemaker
from sagemaker import get_execution_role 
sagemaker_session = sagemaker.Session() 
role = sagemaker.get_execution_role()


# In[6]:


import statsmodels.formula.api as smf


# In[8]:


model = smf.ols('sepalLength ~ sepalwidth + petallength + petalwidth + species',data=df).fit()


# In[9]:


model.summary()


# In[ ]:




