#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#loading the datasets
df = pd.read_csv('Airline.csv')


# In[4]:


df.head()


# In[5]:


# Describing the dataset
df.describe()


# In[6]:


df.shape


# In[7]:


# Information about Each Column
df.info()


# In[8]:


# checking null values
pd.isnull(df).sum()


# In[9]:


df.dropna(inplace=True)


# In[10]:


pd.isnull(df).sum()


# In[11]:


df.duplicated().sum()


# In[12]:


#converting age calumn's value into group
df['Age'].value_counts()


# In[13]:


df['Age Group'] = df['Age'].apply(lambda x: 'Young Age' if x <= 40 else ('Middle Age' if x <= 50 else 'Old Age'))


# In[14]:


df.drop(['Age'], axis=1, inplace=True)


# In[15]:


df


# In[16]:


df.dtypes


# In[17]:


#Dropping the unwanted column
df.drop(['Departure Delay in Minutes','Arrival Delay in Minutes','Gate location','Online boarding','Leg room service'], axis=1, inplace=True)


# In[18]:


df


# # Exploratory Data Analysis - EDA

# In[20]:


df.columns


# In[31]:


#Checking Customer Satisfaction On Online Support
fig, cx = plt.subplots(figsize=(20,5))
sns.lineplot(df,x="Online support",y="satisfaction",palette="muted")


# In[38]:


#Checking Customer Satisfaction On Departure/Arrival time convenient
fig, bx = plt.subplots(figsize=(10,5))
sns.barplot(df,x ="Departure/Arrival time convenient", y ="satisfaction")


# In[47]:


#Checking Flight Distance On Age Group
fig, ax = plt.subplots(figsize=(5,5))
sns.barplot(df, x ="Age Group", y ="Flight Distance")


# In[22]:


#Checking satisfaction On Customer Type
satisf = sns.countplot(data=df,x='satisfaction',hue='Customer Type')
satisf


# In[23]:


#Checking satisfaction On Class
plt.subplots(figsize=(30,10))
sns.lineplot(df,x="Class",y="satisfaction",palette="highlight")


# In[24]:


#Checking satisfaction On Age Group
sns.histplot(df,x="satisfaction",y="Age Group",palette="muted")


# # Correlative Plot

# In[49]:


corr_loan=df.corr()


# In[50]:


corr_loan


# In[53]:


plt.figure(figsize=(10,10))
sns.heatmap(corr_loan,annot=True)


# # Encoding

# In[54]:


from sklearn.preprocessing import LabelEncoder


# In[55]:


label_encoder = LabelEncoder()


# In[56]:


df


# In[57]:


#Replacing value
df['Customer Type'] = label_encoder.fit_transform(df['Customer Type'])
df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])
df['Class'] = label_encoder.fit_transform(df['Class'])
df['Age Group'] = label_encoder.fit_transform(df['Age Group'])
df['satisfaction'] = label_encoder.fit_transform(df['satisfaction'])


# In[58]:


df


# # Train Test Split

# In[59]:


from sklearn.model_selection import train_test_split


# In[60]:


#model building
X=df.drop(columns=['satisfaction',],axis=1)
Y=df['satisfaction']


# In[61]:


X


# In[62]:


Y


# In[63]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=2)


# In[64]:


print(X.shape, X_train.shape, X_test.shape)


# # ML Algorithms

# # Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[66]:


lr = LogisticRegression()
lr.fit(X_train,Y_train)


# In[67]:


pred_lr = lr.predict(X_test)
pred_lr


# In[68]:


acc_lst = accuracy_score(Y_test, pred_lr)
acc_lst


# # Naive byes

# In[69]:


from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB


# In[70]:


sc=StandardScaler()
x_train1=sc.fit_transform(X_train)
x_test1=sc.fit_transform(X_test)


# In[71]:


sc = GaussianNB()
sc.fit(x_train1,Y_train)


# In[72]:


pred_sc=sc.predict(x_test1)
pred_sc


# In[73]:


acc_sc=accuracy_score(Y_test,pred_sc)
acc_sc


# # Random Forest

# In[74]:


from sklearn.ensemble import RandomForestClassifier


# In[75]:


rfc= RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,Y_train)


# In[76]:


pred_rfc = rfc.predict(X_test)
pred_rfc


# In[77]:


acc_rfc= accuracy_score(Y_test,pred_rfc)
acc_rfc


# # Decision Tree

# In[78]:


from sklearn.tree import DecisionTreeClassifier


# In[79]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)


# In[80]:


pred_dtc = dtc.predict(X_test)
pred_dtc


# In[81]:


acc_dtc= accuracy_score(Y_test,pred_dtc)
acc_dtc


# # K-Nearest neighbour

# In[82]:


from sklearn.neighbors import KNeighborsClassifier


# In[83]:


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


# In[84]:


pred_knn = knn.predict(X_test)
pred_knn


# In[85]:


acc_knn = accuracy_score(Y_test, pred_knn)
acc_knn


# # Comparing the best algorithm

# In[86]:


algorithms = ["Logistic Regression","Naive Byes","Random Forest","Decision Tree","K-Nearest"]
scores = [0.7966252220248667,0.8179396092362344,0.9493397173526913,0.9305351764615029,0.8284423507606765] 


# In[89]:


# Creating the bar chart
plt.figure(figsize=(8, 5))  
plt.bar(algorithms, scores, color='skyblue')
plt.xlabel('Algorithms')
plt.ylabel('Score')
plt.title('Machine Learning Algorithm Comparision')
plt.show()


# # Model Prediction

# In[90]:


import joblib as jb


# In[91]:


jb.dump(rfc,'Satisfaction_Predict')


# In[92]:


model=jb.load('Satisfaction_Predict')


# In[93]:


#getting values from df - 1st row for accuracy check
df2=pd.DataFrame({
    'Customer Type':0,
    'Type of Travel':1,
    'Class':1,
    'Flight Distance':265,
    'Seat comfort':0,
    'Departure/Arrival time convenient':0,
    'Food and drink':0,
    'Inflight wifi service':2,
    'Inflight entertainment':4,
    'Online support':2,
    'Ease of Online booking':3,
    'On-board service':3,
    'Baggage handling':3,
    'Checkin service':5,
    'Cleanliness':3,
    'Age Group':1
},index=[0])


# In[94]:


df2


# In[95]:


result=model.predict(df2)


# In[96]:


if result ==1:
    print("Congratulation! Your Customer Is Satisfied")
else:
    print("Sorry... Your Customer Is dissatisfied")


# In[ ]:


#Hence our predicting value and main value are equal/corrrect.

