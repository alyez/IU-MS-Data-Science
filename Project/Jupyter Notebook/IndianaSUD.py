
# coding: utf-8

# # Visualizing high dimensional TEDS data

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

import numpy as np
import scipy.stats as ss

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[7]:


df=pd.read_csv('__TEDSCountyLevel2008-2017_2.csv')


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


df.head(10)


# In[9]:


#Limit the number of features
features = ['Year','County Name','Gender','Race','Age','Alcohol','Heroin','Marijuana','Opioid','Cocaine','Other Drugs','Pain Killers', 'Methamphetamine', 'Synthetic Drugs']
df_subset = df[features]
df_subset.head()


# In[10]:


import altair as alt
alt.renderers.enable('notebook')


# In[11]:


#Create a dataset for the 9 substances to use in correlations
features = ['Alcohol','Heroin','Marijuana','Opioid','Cocaine','Other Drugs','Pain Killers', 'Methamphetamine', 'Synthetic Drugs']
df_only_features = df_subset[features]
df_only_features.head()


# In[12]:


n_samples, n_features = df_only_features.shape
print(n_samples)
print(n_features)


# In[11]:


#Check correlation between substances
import seaborn as sns

f, ax = plt.subplots(figsize=(9, 9))
corr = df_only_features.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0,
            square=True, ax=ax,annot=True)


# There is strong correlation between Pain Killers and Synthetic Drugs, supported by the Opioids. Heroin appears in 57% of the cases with Opioids. Alcohol is slightly negatively correlates with all four of them. 
# 
# #### Remarkably, there appears to be no appreciable correlation between opioid use and such commonly accepted and complementary socio-economic markers as alcohol, marijuana, or methamphetamine use. In fact, these three factors are likely to be biased towards older population (alcohol), younger population (marijuana) and underprivileged rural population (methamphetamine). 
# #### Thus, it appears that opioid abuse have roots that are different from the common socio-economic maladies reflected in these three markers. 
# 

# In[13]:


df_subset['County Name'].nunique()


# In[13]:


df_subset['Year'].nunique()


# In[14]:


# Total number of records
len(df_subset)


# In[15]:


#How many records per year
df_subset['Year'].value_counts()


# In[15]:


#How many records per county 
df_subset['County Name'].value_counts()


# In[16]:


# General method to variably group the data
def group_dataset(df, col1, col2):
    df_grouped=df.groupby([col1,col2])
    return df_grouped


# In[17]:


#Group by Year and County
df_grouped=group_dataset(df_subset, 'Year','County Name')

#Excluded pain killers and synthetic drugs as highly correclated with opioids
feature=['Alcohol','Heroin','Marijuana','Opioid','Cocaine','Other Drugs','Methamphetamine']
df_grouped=df_grouped[feature].mean()

df_grouped.head()


# In[18]:


#Group by County
df_grouped_c=group_dataset(df_subset,'County Name', 'Year')

#exclude pain killers and synthetic drugs as highly correclated with opioids
feature=['Alcohol','Heroin','Marijuana','Opioid','Cocaine','Other Drugs','Methamphetamine']
df_grouped_c=df_grouped_c[feature].mean()

df_grouped_c.head()


# In[19]:


# Group by Year to genralize the output
df_grouped_year1=df_subset.groupby('Year')
df_grouped_year1=df_grouped_year1[feature].mean()

df_grouped_year1.head()


# In[20]:


# General method to pivot by two predictors
def create_pivot_df(df, col_name, index_name, val_name):
    pivot_df = df.pivot_table(columns=col_name, index= index_name, values= val_name, aggfunc='mean')
    return pivot_df


# In[21]:


#create a pivot table for alcohol
create_pivot_df(df_grouped,col_name='Year',index_name='County Name', val_name='Alcohol').head()


# In[22]:


def draw_heatmap(df, nrows, ncols):
    f, ax = plt.subplots(figsize=(nrows, ncols))    
    sns.heatmap(df,  cmap=sns.cubehelix_palette(8, start=.5, rot=-.75),ax=ax,xticklabels=2, square=True, vmin=0, vmax=.9)


# In[23]:


def draw_heatmap_corr(df, nrows, ncols):
    f, ax = plt.subplots(figsize=(nrows, ncols)) 
    df=df.corr()
    sns.heatmap(df,  cmap=sns.diverging_palette(220, 10, as_cmap=True), center=0,ax=ax, square=True,annot=True,cbar_kws={"shrink": .5})  


# In[24]:


#Transpose the matrix
df_grouped_year_T= df_grouped_year1.T
df_grouped_year_T


# In[27]:


# Show a trend relative to other substances
draw_heatmap(df_grouped_year_T, 10, 7)


# In[28]:


# Draw a heatmap by County for Alcohol
draw_heatmap(
create_pivot_df(df_grouped,'Year','County Name', 'Alcohol'), 186, 20)


# In[29]:


#Opioid
draw_heatmap(create_pivot_df(df_grouped,'Year','County Name', 'Opioid'), 186, 20)


# In[30]:


#Marijuana
draw_heatmap(create_pivot_df(df_grouped,'Year','County Name', 'Marijuana'), 186, 20)


# In[31]:


# Methamphetamine
draw_heatmap(create_pivot_df(df_grouped,'Year','County Name', 'Methamphetamine'), 186, 20)


# #### From the four heatmaps above, one could infer the following observations: 
# - while Alcohol remains the most commonly abused substance, its dominance is vaning as the prevalence of other substances, especially opiods and methamphethamine increases. At the same time, marijuana abuse remains fairly stable.  

# ### Suplimentary Analysis: Identify any years that differ from the majority

# In[ ]:


draw_heatmap_corr(create_pivot_df(df_grouped,'Year','County Name', 'Alcohol'), 10,10)


# In[33]:


#Marijuana
draw_heatmap_corr(create_pivot_df(df_grouped,'Year','County Name', 'Marijuana'), 10,10)


# In[34]:


#Opioids
draw_heatmap_corr(create_pivot_df(df_grouped,'Year','County Name', 'Opioid'), 10,10)


# In[35]:


#Methamphetamine
draw_heatmap_corr(create_pivot_df(df_grouped,'Year','County Name', 'Methamphetamine'), 10,10)


# Beginning 2012 the usage patterns became very similar for the malority of the subjects that requested treatment in Indiana.

# ### Suplimentary Analysis: Identify counties that differ from the majority

# In[37]:


draw_heatmap_corr(create_pivot_df(df_grouped,'County Name','Year', 'Alcohol'), 93,93)


# In[38]:


draw_heatmap_corr(create_pivot_df(df_grouped,'County Name','Year', 'Opioid'), 93,93)


# In[39]:


draw_heatmap_corr(create_pivot_df(df_grouped,'County Name','Year', 'Marijuana'), 93,93)


# In[40]:


draw_heatmap_corr(create_pivot_df(df_grouped,'County Name','Year', 'Other Drugs'), 93,93)


# In[41]:


draw_heatmap_corr(create_pivot_df(df_grouped,'County Name','Year', 'Heroin'), 93,93)


# In[42]:


draw_heatmap_corr(create_pivot_df(df_grouped,'County Name','Year', 'Methamphetamine'), 93,93)


# ## Suplimentary analisys: PCA 
# 
# The [principal component analysis (PCA)](http://setosa.io/ev/principal-component-analysis/) is the most basic dimensionality reduction method. To run the PCA we want to isolate only the numerical columns. 

# In[43]:


from sklearn.decomposition import PCA
pca = PCA() 


# Now I ran `fit()` method to identify principal components. 

# In[44]:


pca_df_fitted = pca.fit(df_only_features)
print(pca.components_)


# '-Other Drugs', Marijuana,Opioid+Alcohol, Marijuana+Cocaine, Marijuana+Alcohol, -Marijuana-Heroin, Marijuana+Methamphetamine

# In[45]:


print(pca.explained_variance_)


# In[46]:


pca_df_fitted.explained_variance_ratio_


# In[47]:


pca.components_.shape


# In[48]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# The first six components capture more than 95% of the variance in original dataset. This means that the PCA is not very effective on this dataset and six components will provide approximation for the rest of the dimensions. 
