#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


print(' importing data from s3 bucket')


# In[14]:


bucket= 'bucket-dataset/Regression'
data_key= 'insurance.csv'
data_location='s3://{}/{}'.format(bucket,data_key) 
df=pd.read_csv(data_location)


# In[15]:


df


# In[16]:


df.shape


# In[17]:


df.sex.replace(['male','female'],['1','0'], inplace=True)


# In[18]:


df.smoker.replace(['yes','no'],['1','0'],inplace=True)


# In[19]:


print(df.dtypes)


# In[20]:


df.head()


# In[21]:


x= df.drop(['region','charges'], axis=1)
y=df['charges']


# In[22]:


x


# In[23]:


y


# In[24]:


import numpy as np


# In[25]:


x = x.astype('float32')
y = y.astype('float32')


# In[ ]:


## splitting to testing and training


# In[26]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=4)


# In[27]:


x_train.shape


# In[ ]:





# In[28]:


y_train.shape


# In[29]:


x_test.shape


# In[30]:


y_test.shape


# In[117]:


x_train=x_train.astype('float32')


# In[118]:


y_train=y_train.astype('float32')


# In[13]:


import boto3
import sagemaker 
from sagemaker import get_execution_role

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()


# In[144]:


from sagemaker import LinearLearner
bucket = 'bucket-dataset'
prefix= 'sm-out'
output_path = 's3://{}/{}'.format(bucket,prefix)
linear= LinearLearner(role=role,
                     instance_count=1,
                     instance_type='ml.m4.xlarge',
                     predictor_type='regressor',
                     output_path=output_path,
                     sagemaker_session=sagemaker_session,
                     epochs=300,
                     num_models=32,
                     loss='logistic',
                      use_spot_instances=True,
                     max_run=300,
                      max_wait=600,
                     )


# In[145]:


formatted_train_data = linear.record_set(x_train.values, labels=y_train.values)


# In[146]:


formatted_validation_data = linear.record_set(x_test.values, labels= y_test.values, channel='validation')


# In[147]:


linear.fit([formatted_train_data,formatted_validation_data])


# In[ ]:


linear_regressor= linear.deploy(initial_instance_count=1, instance_type='ml.t2.medium')


# In[12]:


from sagemaker.predictor import CSVSerializer, JSONDeserializer 

linear_regressor.ContentType = 'text/csv'
linear_regressor.serializer = CSVSerializer
linear_deserializer = JSONDeserializer 


# In[9]:


result = linear_regressor.predict(x_test)
result


# In[178]:


import sklearn.metrics as metrices


# In[200]:


print('MSE', metrices.mean_squared_error(x_train,y_train))


# In[187]:


model.summary()


# In[197]:


x_train.shape


# In[198]:


x_test.shape


# In[201]:


y_test.shape


# In[ ]:




