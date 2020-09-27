#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from numpy import genfromtxt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("creditcard.csv")
time = np.reshape(np.array(data['Time']), (-1, 1))
amount = np.reshape(np.array(data['Amount']), (-1, 1))
#Preparing input features
X = np.reshape(np.array(data['V1']), (-1, 1))
for i in range(2, 29):
    col = 'V' + str(i)
    adder = np.reshape(np.array(data[col]), (-1, 1))
    X = np.concatenate([X, adder], axis=1)
y = np.reshape(np.array(data['Class']), (-1, 1))
scaler = StandardScaler()
#Scaling 'amount' and 'time' features
time = scaler.fit_transform(time)
amount = scaler.fit_transform(amount)
X = np.concatenate([X, time, amount], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)
#Using SMOTE to synthetically create examples to even out imbalanced dataset (very few fraud examples)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

#Setting up for neural network here
model = MLPClassifier(alpha=0.001, hidden_layer_sizes=(35), learning_rate_init=0.001, max_iter=500)

#TRAINING THE NETWORK
model.fit(X_train, y_train) # training targets

#PREDICTING WITH THE NETWORK
predict = model.predict(X_test)
print (roc_auc_score(y_test, predict))
print(classification_report(y_test,predict))


# In[12]:





# In[ ]:




