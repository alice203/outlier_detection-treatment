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


# In[9]:


#Scatterplot
ax = sns.scatterplot(x="TAX", y="INDUS", data=df)


# In[5]:


#Mahalonibis Distance
def mahalanobis_method(x=None, data=None):
    x_minus_mu = x - np.mean(data)
    cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = mahal.diagonal()
    #Compare MD with threshold and flag as outlier
    outlier = []
    C = chi2.ppf((1-0.001), df=2)
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier

outliers_mahal = mahalanobis_method(x=df, data=df)
print(len(outliers_mahal))
print(outliers_mahal)


# In[6]:


#Robust Mahalonibis Distance
def robust_mahalanobis_method(x=None, data=None):
    x_minus_mu = x - np.mean(data)
    #Minimum covariance determinant method
    rng = np.random.RandomState(0)
    real_cov = np.cov(data.values.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_
    inv_covmat = sp.linalg.inv(mcd)
    left_term = np.dot(x_minus_mu, inv_covmat)
    #Calculate MD with minimum covariance determinant method
    mahal = np.dot(left_term, x_minus_mu.T)
    md = mahal.diagonal()
    #Compare MD with threshold and flag as outlier
    outlier = []
    C = chi2.ppf((1-0.001), df=2)
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier

outliers_mahal = robust_mahalanobis_method(x=df, data=df)
print(len(outliers_mahal))
print(outliers_mahal)

