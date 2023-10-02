#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math 


# In[37]:


titanic= pd.read_csv(r'C:\Users\Nandita\Downloads\train.csv')


# In[38]:


titanic.head()


# In[40]:


titanic.shape


# In[42]:


sns.countplot(x='Survived',data=titanic)


# In[43]:


titanic


# In[44]:


sns.countplot(x='Survived',hue='Pclass', data=titanic, palette='PuBu')


# In[45]:


titanic['Age'].plot.hist()


# In[10]:


titanic['Fare'].plot.hist(bins=20,figsize=(10,5))


# In[11]:


sns.countplot(x='SibSp',data=titanic,palette='rocket')


# In[13]:


titanic['Parch'].plot.hist()


# In[14]:


sns.countplot(x='Parch',data=titanic,palette='summer')


# In[15]:


titanic.isnull().sum()


# In[16]:


sns.heatmap(titanic.isnull(),cmap='spring')


# In[17]:


sns.boxplot(x='Pclass',y='Age',data=titanic)


# In[18]:


#we can observe that older agegroup are travling more in class 1 and 2
# comapared to class 3
# the hue parameter determines which colums in the datya frame should be used for colour encoding 
#we will drop a frew coloums and  row 
titanic.head()


# In[19]:


#droping the 
titanic.drop('Cabin',axis=1,inplace=True)
titanic.head(3)


# In[20]:


titanic.dropna(inplace=True)
sns.heatmap(titanic.isnull(),cbar=False)


# In[21]:


titanic.isnull().sum()


# In[22]:


titanic.head(2)


# In[23]:


pd.get_dummies(titanic['Sex']).head()


# In[24]:


sex=pd.get_dummies(titanic['Sex'],drop_first=True)
sex.head(3)


# In[25]:


#we have droped the firsst coloum because onnly one coloum is sufficient to deter maine 
#the gender of the passanger 
embark=pd.get_dummies(titanic['Embarked'])
embark.head()


# In[26]:


# C for cherbourd , q for queenstow &S for southhamptom
embark=pd.get_dummies(titanic['Embarked'],drop_first=True)
embark.head(3)


# In[27]:


# C for cherbourd , q for queenstow &S for southhamptom
embark=pd.get_dummies(titanic['Embarked'],drop_first=True)
embark.head(3)


# In[30]:


#our data is now converted in to categorial data
titanic=pd.concat([titanic,embark],axis=1)
titanic.head(3)


# In[31]:


#delleting the unwanted colums 
titanic.drop(['Name','PassengerId','Pclass','Ticket','Sex','Embarked'],axis=1,inplace=True)
titanic.head(3)


# In[32]:


# Train Data
X= titanic.drop('Survived',axis=1)
y=titanic['Survived']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.33,random_state=4)


# In[33]:


from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
lm.fit(X_train, y_train)


# In[34]:


X_test.columns = X_test.columns.astype(str)
prediction=lm.predict(X_test)
from sklearn.metrics import confusion_matrix 
confusion_matrix(y_test,prediction)


# In[35]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)


# In[ ]:


# we have the accuracy od 72 % which is quite good and the model can predict and data quite accuratly

