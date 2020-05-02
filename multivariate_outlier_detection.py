#!/usr/bin/env python
# coding: utf-8

# In[158]:


from sklearn.datasets import load_boston
import copy 
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import scipy as sp
from scipy.stats import chi2
from sklearn.covariance import MinCovDet


# In[137]:


#Load data
boston = load_boston()
X, y = load_boston(return_X_y=True)

#Create data frame
columns = boston.feature_names
df = pd.DataFrame(X, columns = columns)


# In[138]:


df


# In[139]:


df.describe()


# In[140]:


#Scatterplot
ax = sns.scatterplot(x="RM", y="INDUS", data=df)


# In[141]:


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
    C = chi2.ppf((1-0.001), df=x.shape[1]) #degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md

outliers_mahal, md = mahalanobis_method(x=df, data=df)
print(len(outliers_mahal))
print(outliers_mahal)
df['md'] = md


# In[154]:


#Robust Mahalonibis Distance
def robust_mahalanobis_method(x=None, data=None):
    #Minimum covariance determinant method
    rng = np.random.RandomState(0)
    real_cov = np.cov(data.values.T)
    X = rng.multivariate_normal(mean=np.mean(data, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ #robust covariance metric
    robust_mean = cov.location_  #robust mean
    inv_covmat = sp.linalg.inv(mcd) #inverse covariance metric
    
    #Calculate MD with minimum covariance determinant method
    x_minus_mu = x - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = mahal.diagonal()
    
    #Compare rMD with threshold and flag as outlier
    outlier = []
    C = chi2.ppf((1-0.001), df=x.shape[1]) #degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md

outliers_mahal_rob, md_rb = robust_mahalanobis_method(x=df, data=df)
print(len(outliers_mahal_rob))
print(outliers_mahal_rob)
df['md_rob'] = md_rb


# In[166]:


#Visualization
bivariate_df = df[['RM', 'INDUS']]
bv_outliers_mahal,md = mahalanobis_method(x=bivariate_df, data=bivariate_df)
bv_outliers_mahal_rob, md_robust = robust_mahalanobis_method(x=bivariate_df, data=bivariate_df)

def flag_outliers(df, outliers):
    flag = []
    for index in range(df.shape[0]):
        if index in outliers:
            flag.append(1)
        else:
            flag.append(0)
    return flag

flag_outlier_classic = flag_outliers(bivariate_df,bv_outliers_mahal)
bv_df = copy.deepcopy(bivariate_df)
bv_df['flag'] = flag_outlier_classic
bv_df['md'] = md
bv_df['flag_rob'] = flag_outliers(bivariate_df,bv_outliers_mahal_rob )
bv_df['md_robust'] = md_robust


# In[108]:


ax = sns.scatterplot(x="RM", y="INDUS", hue='flag', data=df)


# In[112]:


ax = sns.scatterplot(x="RM", y="INDUS", hue='flag_rob', data=df)


# In[167]:


i = np.arange(0,df.shape[0],1) 
bv_df['index'] = i
ax = sns.scatterplot(x="index", y="md", data=bv_df, hue ='flag')


# In[170]:


ax = sns.scatterplot(x="index", y="md_robust", data=bv_df, hue ='flag_rob')


# In[182]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px

fig1 = px.scatter(bv_df, x='index', y='md', color='flag')
fig1.add_shape(dict(type="line", x0=0, y0=15, x1=500, y1=15, line_dash="dot", 
                   line=dict(color="orange", width=3)))
fig1.update_layout(xaxis_title="index", yaxis_title="Mahalanobis distance")
fig1


# In[183]:



fig2 = px.scatter(bv_df, x='index', y='md_robust', color='flag_rob')
fig2.add_shape(dict(type="line", x0=0, y0=15, x1=500, y1=15, line_dash="dot", 
                   line=dict(color="orange", width=3)))
fig2.update_layout(xaxis_title="index", yaxis_title="Mahalanobis distance")
fig2

