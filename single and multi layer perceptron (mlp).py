#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
Data  = [[0,0,255,"BLUE"], 
        [0, 0, 255,"BLUE"],
        [0, 0, 192, "BLUE"],
        [243, 80, 59, "RED"],
        [255, 0, 77, "RED"],
        [77, 93, 190, "BLUE"],
        [255, 98, 89, "RED"],
        [208, 0, 49, "RED"],
        [67, 15, 210, "BLUE"],
        [82, 117, 174, "BLUE"],
        [168, 42, 89, "RED"],
        [248, 80, 68, "RED"],
        [128, 80, 255, "BLUE"],
        [228, 105, 116, "RED"]]
dataframe= pd.DataFrame(Data, columns = ['R', 'G', 'B','CLASSIFICATION'])
dataframe


# In[2]:


import numpy as np


dataframe.info()

x=dataframe.iloc[:,:3]
print(x)
y=dataframe.iloc[:,3]
dataframe['CLASSIFICATION']=dataframe['CLASSIFICATION'].replace("BLUE","0")
dataframe['CLASSIFICATION']=dataframe['CLASSIFICATION'].replace("RED","1")
print(y)


# In[3]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
minmax_dataframe = scaler.fit_transform(x)
minmax_dataframe = pd.DataFrame(minmax_dataframe, columns =['R','G','B'])
x


# In[4]:


label_encoder = preprocessing.LabelEncoder() 
dataframe['CLASSIFICATION']= label_encoder.fit_transform(dataframe['CLASSIFICATION'])
dataframe['CLASSIFICATION'].unique()
y=dataframe['CLASSIFICATION']

dataframe.info()


# In[5]:


from sklearn.model_selection import train_test_split
trainx , testx = train_test_split (x , test_size = 0.25)
trainy , testy = train_test_split (y , test_size = 0.25)


# In[6]:


from sklearn.linear_model import Perceptron
model = Perceptron(max_iter=1800)
z=model.fit(trainx , trainy)
ypred = model.predict(testx)


# In[7]:


from sklearn.metrics import accuracy_score
print(accuracy_score(testy,ypred))


# In[8]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(testy,ypred))
rmse = np.sqrt(mean_squared_error(testy,ypred))
print(rmse)


# In[ ]:




