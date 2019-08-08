#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("batterylog.txt")


# In[25]:


df.head()
df.shape


# In[6]:


plt.scatter(df.c_time, df.l_time, color = 'red')
plt.xlabel("Charging Time")
plt.ylabel("Lasting Time")
plt.show()


# In[40]:


from sklearn import linear_model
dff = df[df['c_time']<4]


# In[35]:


regr = linear_model.LinearRegression()
train_x = np.asanyarray(dff[['c_time']])
train_y = np.asanyarray(dff[['l_time']])
regr.fit(train_x,train_y)
plt.scatter(df.c_time,df.l_time,color = 'blue')
plt.plot(train_x,regr.coef_[0][0]*dff.c_time + regr.intercept_[0],color = 'red')
plt.show()


# In[36]:


print("Coef: ",regr.coef_)
print("Intercept: ",regr.intercept_)


# In[39]:


test = float(input())
coef = int(regr.coef_)
inter = int(regr.intercept_)
y = min((coef*test + inter),8.00)
print(y)


# In[ ]:




