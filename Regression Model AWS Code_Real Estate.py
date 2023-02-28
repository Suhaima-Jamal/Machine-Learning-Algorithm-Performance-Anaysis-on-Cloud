#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd


# In[ ]:


import numpy as np


# In[60]:


bucket= 'bucket-dataset/Regression'
data_key= 'Real estate.csv'
data_location= 's3://{}/{}'.format(bucket,data_key) 
df=pd.read_csv(data_location)


# In[61]:


df


# In[ ]:


df.head()


# In[62]:


df.shape


# In[ ]:


x = x.astype('float32')
y = y.astype('float32')


# In[63]:


from sklearn.model_selection import train_test_split 
training_data = df.sample(frac=0.7, random_state=25) 
testing_data= df.drop(training_data.index)
print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# In[64]:


training_data


# In[65]:


testing_data


# In[66]:


import boto3
import sagemaker
from sagemaker import get_execution_role 
sagemaker_session = sagemaker.Session() 
role = sagemaker.get_execution_role()


# In[67]:


import statsmodels.formula.api as smf


# In[68]:


model=smf.ols('unitAreaPrice ~ x1 + x2 + x3 + x4 + x5 +x6',data = df).fit()


# In[69]:


model.summary()


# In[ ]:




