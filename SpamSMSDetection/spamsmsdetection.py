# -*- coding: utf-8 -*-
"""SpamSMSDetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gwV0_oToEH0vt_s7y3uMkMrBo0xPbL2F

Importing Modules/Dependencies for Data Handling
"""

import numpy as np
import pandas as pd

"""Loading the data set downloaded from Kaggle : https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data"""

# loading the dataset
data = pd.read_csv('/content/spam.csv', encoding='ISO-8859-1')

"""Data Preprocessing :"""

print(data)

# replace the null values with a null string
data = data.where((pd.notnull(raw_mail_data)),'')

# removing empty columns
data = data.iloc[:, :2]

# printing the first 5 rows of the dataframe
data.head()

"""V1 - Classification
<br>
V2 - Message
"""

# checking the number of rows and columns in the dataframe
data.shape

# label spam mail as 0;  ham mail as 1;
data.loc[data['v1'] == 'spam', 'v1',] = 0
data.loc[data['v1'] == 'ham', 'v1',] = 1

# separating the data as texts (inputs) and label(target)
X = data['v2']
Y = data['v1']

print(X)

print(Y)

"""Splitting the data set into Training data and Testing data"""

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""Creating a TfidfVectorizer object :
<br>
It is a numerical statistic that reflects how important a word is to a document in a collection or corpus. It's a product of two metrics: Term Frequency (TF) and Inverse Document Frequency (IDF).
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# transform the text data to feature vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)

print(X_train_features)

"""Importing the Dependencies of the Required ML Models to Experiment on the given data set:"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

model_lr = LogisticRegression() #Logistic Regression Classification Model

# training the Logistic Regression model with the training data
model_lr.fit(X_train_features, Y_train)

# prediction on training data
prediction_on_training_data = model_lr.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# prediction on test data
prediction_on_test_data = model_lr.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

model_nb = MultinomialNB() # Naive Bayes Classification Model

# training the Logistic Regression model with the training data
model_nb.fit(X_train_features, Y_train)

# prediction on training data
prediction_on_training_data = model_nb.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# prediction on test data
prediction_on_test_data = model_nb.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

model_svc = SVC(kernel='linear') # Support Vector Classification Model

# training the Logistic Regression model with the training data
model_svc.fit(X_train_features, Y_train)

# prediction on training data
prediction_on_training_data = model_svc.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print('Accuracy on training data : ', accuracy_on_training_data)

# prediction on test data
prediction_on_test_data = model_svc.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

"""Thus, Support Vector Classifier has the best accuracy score over training data of 99.73%

Testing the Model :
"""

input_mail = ["Hurray!! You won the million dollar lottery, call this Nuber 56783894 and share your bank data to win the price 450000"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction using SVC Model
prediction = model_svc.predict(input_data_features)
print(prediction)

if (prediction[0]==1):
  print('Legitimate SMS')
else:
  print('Spam SMS')

input_mail = ["Free entry in 2 a wkly comp to win FA Cup"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction
prediction = model_svc.predict(input_data_features)
print(prediction)

if (prediction[0]==1):
  print('Legitimate SMS')
else:
  print('Spam SMS')