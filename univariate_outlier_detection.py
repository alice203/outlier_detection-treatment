#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy as sp
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


# In[2]:


#Load data
boston = load_boston()
X, y = load_boston(return_X_y=True)

#Create data frame
columns = boston.feature_names
df = pd.DataFrame(X, columns = columns)


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


ax = sns.boxplot(x=df["RM"])


# In[6]:


#Tukey's method
def tukeys_method(variable):
    q1 = df["CRIM"].quantile(0.25)
    q3 = df["CRIM"].quantile(0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr
    #inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence
    #outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    outliers = []
    for index, x in enumerate(variable):
        if x <= outer_fence_le or x >= outer_fence_ue:
            outliers.append(index)
    return outliers
        
probable_outliers_cr = tukeys_method(df["CRIM"])
print(probable_outliers_cr)
probable_outliers_zn = tukeys_method(df["ZN"])
print(probable_outliers_zn)


# In[7]:


#Internally studentized method (z-score)
def z_score_method(df, variable_name):
    columns = df.columns
    z = np.abs(stats.zscore(df))
    threshold = 3
    outlier = []
    index=0
    for item in range(len(columns)):
        if columns[item] == variable_name:
            index == item
    for i, v in enumerate(z[:, index]):
        if v > threshold:
            outlier.append(i)
        else:
            continue
    return outlier

outlier_z = z_score_method(df, 'ZN')
print(outlier_z)


# In[8]:


#MAD method
def mad_method(df, variable_name):
    columns = df.columns
    med = np.median(df, axis = 0)
    mad = np.abs(stats.median_absolute_deviation(df))
    threshold = 3
    outlier = []
    index=0
    for item in range(len(columns)):
        if columns[item] == variable_name:
            index == item
    for i, v in enumerate(df.loc[:,variable_name]):
        t = (v-med[item])/mad[item]
        if t > threshold:
            outlier.append(i)
        else:
            continue
    return outlier

outlier_mad = mad_method(df, 'ZN')
print(outlier_mad)

