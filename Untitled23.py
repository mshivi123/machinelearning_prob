
# coding: utf-8

# In[1]:


from sklearn import datasets


# In[2]:


mydata=datasets.load_iris()


# In[3]:


mydata.keys()


# In[4]:


x_input=mydata.data


# In[5]:


y_input=mydata.target


# In[7]:


from sklearn.model_selection import train_test_split


# In[12]:


x_train,x_test,y_train,y_target=train_test_split(x_input,y_input,test_size=0.3)


# In[13]:


y_train


# In[16]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


# In[17]:


myber=BernoulliNB()
mygau=GaussianNB()
mymul=MultinomialNB()


# In[19]:


mygaumodel=mygau.fit(x_train,y_train)
mybermodel=myber.fit(x_train,y_train)
mymulmodel=mymul.fit(x_train,y_train)


# In[20]:


ypgau=mygaumodel.predict(x_test)
ypber=mybermodel.predict(x_test)
ypmul=mymulmodel.predict(x_test)


# In[21]:


from sklearn import metrics


# In[24]:


acc_gau=metrics.accuracy_score(y_target,ypgau)
acc_ber=metrics.accuracy_score(y_target,ypber)
acc_mul=metrics.accuracy_score(y_target,ypmul)

