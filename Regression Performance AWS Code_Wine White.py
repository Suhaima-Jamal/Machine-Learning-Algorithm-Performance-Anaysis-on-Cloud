#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[ ]:


import numpy as np


# In[2]:


bucket= 'bucket-dataset/Regression'
data_key= 'winequality-white.csv'
data_location= 's3://{}/{}'.format(bucket,data_key) 
df=pd.read_csv(data_location)


# In[ ]:


x = x.astype('float32')
y = y.astype('float32')


# In[3]:


df


# In[ ]:


df.shape


# In[ ]:


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


y_train.shape


# In[ ]:


x_test.shape


# In[ ]:


y_test.shape


# In[ ]:


training_data


# In[ ]:


testing_data


# In[10]:


import boto3
import sagemaker
from sagemaker import get_execution_role 
sagemaker_session = sagemaker.Session() 
role = sagemaker.get_execution_role()


# In[6]:


import statsmodels.formula.api as smf


# In[12]:


model=smf.ols('quality ~ fixedAcidity + volatileAcidity + citricAcid + residualSugar + chlorides + freeSulfurDioxide + totalSulfurDioxide + totalSulfurDioxide + density + pH + sulphates + alcohol',data = df).fit()


# In[9]:


model.summary()


# In[ ]:




