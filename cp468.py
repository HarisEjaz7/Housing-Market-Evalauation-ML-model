#!/usr/bin/env python
# coding: utf-8

# In[124]:


#import initial packages in
import pandas as pd
import numpy as np 


# In[125]:


#Read dataset into python
housing  = pd.read_csv("housing.csv")


# In[126]:


#Print dataset to see different variables
housing


# In[128]:


housing.drop(columns= ["address"], inplace= True)


# In[129]:


print(housing.shape)
housing.head()


# In[130]:


#check dataset for null values and remove any
empty_entries = housing.isnull().sum()

print(empty_entries)


# In[131]:


#reduce outliers of dataset for better model
housing = housing[(housing.pricem >= 0.5) & (housing.pricem <= 2)]




# In[132]:


housing


# In[133]:


#reset index to match record count
housing = housing.reset_index().drop(columns= ["index"])
housing


# In[110]:


#verify the number of Non-null data is equal for all fields
housing.info()


# In[134]:


#Using scikit-learn we are able to split data 
#Convert citys into 0-1 reigon area codes
from sklearn.model_selection import train_test_split
x = pd.get_dummies(housing.drop(columns= ["pricem", 'price']), prefix= 'region', dtype = int)
y = housing['pricem']
x


# In[135]:


#Split train-test data 70-30 split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)


# In[136]:


#Scale dataset and perform linear regression on it
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_train)
reg = LinearRegression()
reg.fit(x_scaled, y_train)
rval = reg.score(x_scaled, y_train)
print("R-squared score (Linear Regression):", rval)


# In[140]:


#calculate the mean absoloute error of the dataset to measure performance
from sklearn.metrics import mean_absolute_error
train_predict = reg.predict(x_scaled)
err_train = mean_absolute_error(y_train, train_predict)
print("Mean absolute error for training:", err_train)
test_scaled = scaler.transform(x_test)
test_predict = reg.predict(test_scaled)
err_test = mean_absolute_error(y_test, test_predict)
print("Mean absolute error for test:", err_test)


# In[138]:


#Plot labelled output against predicted output
import matplotlib.pyplot as plt 
fig = plt.figure()
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  
plt.scatter(y_test,test_predict_poly)
fig.suptitle('Actual vs Predicted', fontsize=20)         
plt.xlabel('Actual', fontsize=18)                  
plt.ylabel('Predicted', fontsize=16)                         


# In[139]:


#Print table to visualize the result using pandas dataframe
import pandas as pd
result_df = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': test_predict})
result_df['Difference'] = result_df['Actual Price'] - result_df['Predicted Price']
print(result_df)


# In[ ]:




