# Credit Fraud Detection Model

This project focuses on building a machine learning model for credit fraud detection using a dataset obtained from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data).

## Introduction

The goal of this project is to develop a model that can effectively identify fraudulent credit card transactions. The dataset contains various features related to credit card transactions, including transaction details, location information, and personal details.

## Data Preprocessing

The initial steps involve loading the training and testing datasets, and then performing data preprocessing. The preprocessing steps include:

- Removing unnecessary columns.
- Converting date-time features to numerical values.
- Creating new features, such as age, month, and year.
- Using one-hot encoding for categorical features.
- Calculating distances between latitude and longitude.

## Exploratory Data Analysis (EDA)

The project includes an exploration of the data to understand its characteristics. This involves creating a correlation matrix to visualize relationships between different features. The heatmap of the correlation matrix helps identify the importance of each feature for prediction.

## Data Sampling

To handle class imbalance, a new dataset is created by combining a sample of legitimate transactions with the entire set of fraudulent transactions. This balanced dataset is used for training and testing the machine learning models.

## Machine Learning Models

Three supervised classification models are implemented:

1. Logistic Regression
2. Decision Tree
3. Random Forest

These models are trained on the preprocessed dataset and evaluated using the testing data. The performance metrics, such as accuracy and classification reports, are then presented.

## Conclusion

The results of the machine learning models are analyzed, and the model with the best performance for credit fraud detection is identified. The project aims to provide insights into the effectiveness of different classification algorithms in identifying fraudulent transactions.

## Usage

1. Ensure the required libraries are installed by running `pip install -r requirements.txt`.
2. Open the Jupyter notebook `CreditFraud.ipynb` in a Jupyter environment.
3. Execute the cells sequentially to load data, preprocess it, perform exploratory data analysis, and train machine learning models.
4. Analyze the results and choose the most suitable model for credit fraud detection.


Author : Md Yasir Khan
