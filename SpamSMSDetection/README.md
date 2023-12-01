# Spam SMS Detection

## Overview

This project aims to detect spam SMS messages using machine learning classification models. The primary focus is on exploring and comparing the performance of different models such as Logistic Regression, Naive Bayes, and Support Vector Classification (SVC).

The project is implemented in a Jupyter Notebook named [SpamSMSDetection.ipynb](SpamSMSDetection.ipynb), which was created using Google Colab. The original notebook can be accessed [here](https://colab.research.google.com/drive/1gwV0_oToEH0vt_s7y3uMkMrBo0xPbL2F).

## Dataset

The dataset used for this project was downloaded from Kaggle: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data). The dataset contains SMS messages labeled as spam or ham (non-spam).

## Data Preprocessing

The notebook includes thorough data preprocessing steps, such as handling null values, removing empty columns, and labeling spam/ham messages. The SMS messages are then split into training and testing sets.

## Feature Extraction

Text data is transformed into feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This vectorizer converts the SMS messages into a numerical format suitable for input to machine learning models.

## Machine Learning Models

Three machine learning classification models are implemented and compared:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Classification (SVC)

The models are trained on the training set, and their accuracies are evaluated on both the training and testing sets.

## Model Testing

The trained models are tested on sample SMS messages to demonstrate their predictive capabilities. Input SMS messages are converted into feature vectors using the TF-IDF vectorizer, and the models predict whether they are spam or ham.

## Results

The accuracy scores and performance metrics of each model are presented in the notebook. The Support Vector Classifier (SVC) achieved the highest accuracy among the tested models.

## How to Use

To run the notebook and experiment with the models, follow these steps:

1. Open the [SpamSMSDetection.ipynb](SpamSMSDetection.ipynb) notebook in Google Colab.
2. Execute the cells in order to load the dataset, preprocess the data, train the models, and make predictions.

Feel free to customize the notebook or extend the project based on your requirements.

# Spam SMS Detection

## Overview

This project aims to detect spam SMS messages using machine learning classification models. The primary focus is on exploring and comparing the performance of different models such as Logistic Regression, Naive Bayes, and Support Vector Classification (SVC).

The project is implemented in a Jupyter Notebook named [SpamSMSDetection.ipynb](SpamSMSDetection.ipynb), which was created using Google Colab. The original notebook can be accessed [here](https://colab.research.google.com/drive/1gwV0_oToEH0vt_s7y3uMkMrBo0xPbL2F).

## Dataset

The dataset used for this project was downloaded from Kaggle: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data). The dataset contains SMS messages labeled as spam or ham (non-spam).

## Data Preprocessing

The notebook includes thorough data preprocessing steps, such as handling null values, removing empty columns, and labeling spam/ham messages. The SMS messages are then split into training and testing sets.

## Feature Extraction

Text data is transformed into feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. This vectorizer converts the SMS messages into a numerical format suitable for input to machine learning models.

## Machine Learning Models

Three machine learning classification models are implemented and compared:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Classification (SVC)

The models are trained on the training set, and their accuracies are evaluated on both the training and testing sets.

## Model Testing

The trained models are tested on sample SMS messages to demonstrate their predictive capabilities. Input SMS messages are converted into feature vectors using the TF-IDF vectorizer, and the models predict whether they are spam or ham.

## Results

The accuracy scores and performance metrics of each model are presented in the notebook. The Support Vector Classifier (SVC) achieved the highest accuracy among the tested models.

## How to Use

To run the notebook and experiment with the models, follow these steps:

1. Open the [SpamSMSDetection.ipynb](SpamSMSDetection.ipynb) notebook in Google Colab.
2. Execute the cells in order to load the dataset, preprocess the data, train the models, and make predictions.

Feel free to customize the notebook or extend the project based on your requirements.

