#!/usr/bin/env python
# coding: utf-8

# In[1]:

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

#reading the csv file and creating as dataframe
df = pd.read_csv('diabetes.csv')
print(df)


# In[2]:


#information of rows, columns, datatype and missing values
df.info()


# In[3]:


#statistics of data
df.describe()


# In[4]:


#correlation matrix
df.corr()


# In[5]:


#plotting correlation matrix
plt.figure(figsize=(8, 8))
sb.heatmap(df.corr(),vmin=-1, vmax=1, center=0, annot=True, cmap='RdBu')


# In[6]:


#creating independent 
Independent = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
x = df[Independent]


# In[7]:


print(x)


# In[8]:

#creating dependent variable outcome which is our final target variable
y = df.Outcome


# In[9]:


print(y)


# In[ ]:





# In[10]:

#importing libraries for classifying our model
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from graphviz import Source
import sklearn.metrics as metrics
import numpy as np


# In[11]:


#splitting the data 67% for training and 33% for testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.33, random_state = 1 )


# In[12]:


#creating decision tree with max depth of 3
clf = DecisionTreeClassifier(criterion="gini", max_leaf_nodes=10, min_samples_leaf=5, max_depth= 3)

#train decision tree
clf = clf.fit(X_train, Y_train)

#predict the model
y_Pred = clf.predict(X_test)
print(y_Pred)


# In[13]:


print("Accuracy is ", metrics.accuracy_score(Y_test, y_Pred))


# In[14]:


dot_data_exp = tree.export_graphviz(clf, out_file=None, feature_names=Independent, class_names=['0','1'], filled= True, rounded= True, special_characters= True )

graph = Source(dot_data_exp)
graph.render('df')
graph.view()


# In[15]:


# visualizing the tree
graph = Source(dot_data_exp)
graph.render('df')
graph.view()


# In[16]:


#training ree
bc_tree = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train,Y_train)


# In[17]:


bc_pred = bc_tree.predict(X_test)


# In[18]:

#predicting the accuracy
bc_tree.score(X_test, Y_test)


# In[19]:

#confusion matrix
cm = confusion_matrix (Y_test, bc_pred)
print(cm)


# In[ ]:





# In[21]:

#analyzing confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the Classifier')
fig.colorbar(cax)
ax.set_xticklabels(['a'])
ax.set_yticklabels(['b'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show


# In[ ]:




