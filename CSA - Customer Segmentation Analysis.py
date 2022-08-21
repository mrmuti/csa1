#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df = pd.read_csv('customers.csv')


# In[6]:


#showing top 5 of the Dataset
df.head(5)


# In[8]:


#showing last 5 of the Dataset
df.tail(5)


# In[14]:


#shape of the Dataset (rows and columns numbers(as count))
def shapes():
    print("number of rows:",df.shape[0])
    print("number of columns:",df.shape[1])
    
data_shapes = shapes()
data_shapes


# In[16]:


#null values in the Dataset

df.isnull().sum()


# In[20]:


#overall static explanations of Dataset
df.describe()


# ## inicate k-means clustering

# In[25]:


from sklearn.cluster import KMeans


# In[27]:


#trying to calculate income per customer under each spending score
random_x = df[['Annual Income (k$)','Spending Score (1-100)']]
random_x


# In[30]:


k_means = KMeans() #press shift+tab to reach more information
k_means.fit(random_x)


# In[33]:


k_means.fit_predict(random_x) #creating depending variable/clusters


# #### lets find optimal number of clusters

# In[37]:


wcss=[]
for i in range(1,11):
    k_means = KMeans(n_clusters=i)
    k_means.fit(random_x)
    wcss.append(k_means.inertia_)
    
wcss #WCSS is the sum of squared distance between each point and the centroid in a cluster. 
     #When we plot the WCSS with the K value, the plot looks like an Elbow. 
     #As the number of clusters increases,the WCSS value will start to decrease.
     


# In[39]:


import matplotlib.pyplot as plt


# In[43]:


plt.plot(range(1,11),wcss)

plt.title('Elbow')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#as you see graphs decreasing strongly up to between 4-6. after that graph decreasing slowly.
#we can consider as elbow that poing. 


# In[48]:


KMeans(n_clusters = 5, random_state=42)
random_y = k_means.fit_predict(random_x)
random_y


# In[64]:


plt.scatter(random_x.iloc[random_y==0,0],random_x.iloc[random_y==0,1],s=100,c='red',label='Cluster0')
plt.scatter(random_x.iloc[random_y==1,0],random_x.iloc[random_y==1,1],s=100,c='black',label='Cluster1')
plt.scatter(random_x.iloc[random_y==2,0],random_x.iloc[random_y==2,1],s=100,c='blue',label='Cluster2')
plt.scatter(random_x.iloc[random_y==3,0],random_x.iloc[random_y==3,1],s=100,c='orange',label='Cluster3')
plt.scatter(random_x.iloc[random_y==4,0],random_x.iloc[random_y==4,1],s=100,c='yellow',label='Cluster4')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100,c='magenta')
plt.legend()
plt.title('CSA')
plt.xlabel('Annuel Income')
plt.ylabel('Spending Score')
plt.show()


# In[66]:


import joblib


# In[68]:


joblib.dump(k_means,'CSA')


# In[70]:


model1 = joblib.load('CSA')


# In[73]:


model1.predict([[15,40]])


# #### Strateji ekibinin ürün grubu ve pazarlama ile ilgili stratejik kararlar alması daha kolay hale geldi.
