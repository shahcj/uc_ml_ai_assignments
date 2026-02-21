# Bank Term Deposit Prediction
## Machine Learning Classification Project

## Business Problem

The goal of this project is to predict whether a bank client will subscribe to a term deposit based on client demographics and marketing campaign data. 

Accurate predictions allow the bank to optimize telemarketing efforts, reduce unnecessary calls, and focus on customers with a higher probability of conversion.

## Project Overview

- The dataset contains 41,188 records.
- The target variable is highly imbalanced (~89% no subscription, ~11% subscription).
- Features include categorical client attributes and marketing campaign variables.

Only bank client features were used for the baseline model:

- Age
- Job
- Marital
- Education
- Default
- Housing
- Loan

Categorical variables were encoded using one-hot encoding. The target variable was converted into binary format.

## Model Comparison

The following models were trained using default parameters:

- Logistic Regression
- KNN
- Decision Tree
- SVM

## Hyperparameter Tuning

GridSearchCV was applied to optimize model parameters using F1-score as the evaluation metric due to class imbalance.

Tuning improved model performance slightly but overall F1 scores remained limited due to imbalance and limited feature signal.'

The confusion matrix shows that models struggle to correctly classify the minority class, reinforcing the impact of class imbalance.

## Conclustion

1. Class imbalance significantly affects model performance.
2. Bank client features alone provide limited predictive power.
3. Models perform better at predicting non-subscribers than subscribers.
