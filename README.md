# NeuroNexus Internship - Machine Learning Projects

## 1. Spam SMS Detection

### Overview

This project focuses on detecting spam SMS messages using machine learning classification models, including Logistic Regression, Naive Bayes, and Support Vector Classification (SVC). The Jupyter Notebook for this project can be found [here](SpamSMSDetection.ipynb).

### Dataset

The dataset used is the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data) obtained from Kaggle, containing labeled SMS messages as spam or ham (non-spam).

### Data Preprocessing

The notebook includes comprehensive data preprocessing steps, such as handling null values, removing empty columns, and labeling spam/ham messages. The SMS messages are split into training and testing sets.

### Feature Extraction

Text data is transformed into feature vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer, enabling numerical input for machine learning models.

### Machine Learning Models

Three classification models are implemented and compared:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Classification (SVC)

Models are trained on the training set, and their accuracies are evaluated on both training and testing sets.

### Model Testing

Trained models are tested on sample SMS messages to demonstrate their predictive capabilities. Input SMS messages are converted into feature vectors using the TF-IDF vectorizer.

### Results

Accuracy scores and performance metrics are presented in the notebook. The Support Vector Classifier (SVC) achieved the highest accuracy among tested models.

### How to Use

1. Open [SpamSMSDetection.ipynb](SpamSMSDetection.ipynb) in Google Colab.
2. Execute cells to load data, preprocess it, train models, and make predictions.

## 2. Credit Fraud Detection Model

### Introduction

This project aims to build a credit fraud detection model using a dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data).

### Data Preprocessing

Steps include loading training and testing datasets, removing unnecessary columns, converting date-time features, creating new features, and handling class imbalance.

### Exploratory Data Analysis (EDA)

Exploration of data involves creating a correlation matrix to visualize relationships between different features.

### Data Sampling

To address class imbalance, a balanced dataset is created by combining a sample of legitimate transactions with all fraudulent transactions.

### Machine Learning Models

Three supervised classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Models are trained on the preprocessed dataset and evaluated using testing data.

### Conclusion

The results are analyzed to identify the most effective model for credit fraud detection.

### Usage

1. Install required libraries with `pip install -r requirements.txt`.
2. Open `CreditFraud.ipynb` in a Jupyter environment.
3. Execute cells sequentially for data loading, preprocessing, exploratory data analysis, and model training.
4. Analyze results to choose the best model for credit fraud detection.
