# -*- coding: utf-8 -*-
"""CreditFraud.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TvEVJB83XFCm3YGNEUMg_N1mxzYyaf7U

Importing Modules/Dependencies for Data Handling
"""

import numpy as np
import pandas as pd

"""Reading Training and Testing Data Sets (Downloaded from https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)"""

credit_data = pd.read_csv('/content/drive/MyDrive/ML_Models/CreditFraud.csv/fraudTrain.csv')
test_data = pd.read_csv('/content/drive/MyDrive/ML_Models/CreditFraud.csv/fraudTest.csv')

# first 5 rows of the dataset
credit_data.head()

# Getting info on the data n the data set
credit_data.info()

"""Creating a function for Data Prerocessing"""

def preprocessing(data) :
  # deleting useless columns
  del_col = ['merchant','first','last','street','zip','unix_time','Unnamed: 0','trans_num','cc_num']
  data.drop(columns=del_col,inplace=True)

  # converting data-time features from object type to Numerical value
  data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
  data['trans_date'] = data['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
  data['trans_date'] = pd.to_datetime(data['trans_date'])
  data['dob'] = pd.to_datetime(data['dob'])

  data["age"] = (data["trans_date"] - data["dob"]).dt.days

  data['trans_month'] = data['trans_date'].dt.month
  data['trans_year'] = data['trans_date'].dt.year

  # using one-hot encoding for categorical data features
  data['gender'] = data['gender'].apply(lambda x : 1 if x=='M' else 0)
  data['gender'] = data['gender'].astype(int)
  data['lat_dis'] = abs(data['lat']-data['merch_lat'])
  data['long_dis'] = abs(data['long']-data['merch_long'])
  data = pd.get_dummies(data,columns=['category'])
  data = data.drop(columns=['city','trans_date_trans_time','state','job','merch_lat','merch_long','lat','long','dob','trans_date'])

  # returning the preprocessed dataset
  return data

# performing data preprocessing on credit_data (Training Data) and test_data
credit_data = preprocessing(credit_data.copy())
test_data = preprocessing(test_data.copy())

# Checking data after preprocessing
credit_data.head()

# Checking the data features, every feature is in numeric data type
credit_data.info()

# creating correlation matrix
correlation_matrix = credit_data.corr()

"""Importing Dependencies for Plotting the Correlation Matrix"""

import seaborn as sns
import matplotlib.pyplot as plt

# plotting the correation matrix
plt.figure(figsize=(14, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Observation : All the data features have low correlation making every fature less and equally important for the prediction,
# except the amount of transaction (amt)

# Counting the number of legit and fraud transaction data in dataset
credit_data['is_fraud'].value_counts()

# separating the data for analysis
legit = credit_data[credit_data.is_fraud == 0]
fraud = credit_data[credit_data.is_fraud == 1]

# priting their shape
print(legit.shape)
print(fraud.shape)

# statistical measures of the legit transaction data
legit.amt.describe()

# statistical measures of the frad transaction data
fraud.amt.describe()

# compare the values for both transactions
credit_data.groupby('is_fraud').mean()

# creating a sample legit data set of same size as that of fraud dataset
legit_sample = legit.sample(n=7506)

# joining the legit and fraud data set
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# checking the new dataset
new_dataset.head()

new_dataset.tail()

# checking the count of legit and fraud transaction datasets in new dataset
new_dataset['is_fraud'].value_counts()

# separating the target and features
X_train = new_dataset.drop(columns='is_fraud', axis=1)
Y_train = new_dataset['is_fraud']

X_test = test_data.drop(columns='is_fraud', axis=1)
Y_test = test_data['is_fraud']

# printing the features
print(X_train)

# printing the target
print(Y_train)

print(X_test)

"""Importing the Dependencies for different Machine Learning Supervised - Classification Models:"""

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# performing feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# performing logistic regression
logistic_regression=LogisticRegression()
logistic_regression.fit(X_train,Y_train)
y_pred_logistic = logistic_regression.predict(X_test)
accuracy_logistic = accuracy_score(Y_test, y_pred_logistic)
accuracy_logistic

DecisionTree=DecisionTreeClassifier()
DecisionTree.fit(X_train,Y_train)
y_pred_dt = DecisionTree.predict(X_test)
accuracy_dt = accuracy_score(Y_test, y_pred_dt)
accuracy_dt

random_forest = RandomForestClassifier(random_state=42,n_estimators=100)
random_forest.fit(X_train, Y_train)
y_pred_rf = random_forest.predict(X_test)
accuracy_rf = accuracy_score(Y_test, y_pred_rf)
accuracy_rf

print("\nClassification Report for Logistic Regression:\n", classification_report(Y_test, y_pred_logistic))
print("\nClassification Report for Decision Tree:\n", classification_report(Y_test, y_pred_dt))
print("\nClassification Report for Random Forest:\n", classification_report(Y_test, y_pred_rf))