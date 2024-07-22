# credit-risk-classification
# Background

In this challenge, a supervised machine learning model was used to categorize the healthy loans and risky loans based on the dataset of historical lending activity from a peer-to-peer lending services company to identify the creditworthiness of borrowers.

Classification is a method of machine learning to predict discrete valued variables. A classification report holds the test results so we can assess and evaluate the number of predicted occurrences for each class.

TP = True Positive, 
TN = True Negative, 
FP = False Positive, 
FN = False Negative

Accuracy, precision, and recall are especially important for classification models that involve a binary decision problem.

-Accuracy is how often the model is correct—the ratio of correctly predicted observations to the total number of observations.

accuracy = (TP + TN) / (TP + TN + FP + FN)

-Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

precision = TP / (TP + FP)

-Recall is the ratio of correctly predicted positive observations to all predicted observations for that class.

recall = TP / (TP + FN)

Logistic regression is a statistical method for predicting binary outcomes from data which is applied to this analysis to perform the classification. 

![image](https://github.com/user-attachments/assets/a73ac3ae-59e2-44f6-b849-f0fee4e7c8b9)

The steps in Logistic Regression Modeling includes Preprocess, Train, Validate, and Predict.

# Table of Contents

1. The Data was split into Training and Testing Sets

2. A Logistic Regression Model with the Original Data was created and evaluated

3. A Credit Risk Analysis Report

# Instructions

Preprocessing the Data

The lending_data.csv file was loaded from the Resources folder into a Pandas DataFrame.

Created the labels set (y) from the “loan_status” column, and then created the features (X) DataFrame from the remaining columns.

NOTE
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

The data was split into training and testing datasets by using train_test_split.

Created a Logistic Regression Model with the Original Data:

Fit a logistic regression model by using the training data (X_train and y_train).

Saved the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance:

- Generate a confusion matrix.

- Print the classification report.

# Report

![image](https://github.com/user-attachments/assets/a942c134-fc43-409f-b81e-59fe16f3c54f)

The purpose: 

To build a classification model of logistic regression for categorizing the credit risk based on the historical dataset to predict the creditworthiness of borrowers.

The results: 

* Accurary of the analysis 99%

* Precision of healthy loans ("0") 100%
  Precision of risky loans ("1") 84%

* Recall of healthy loans ("0") 99%
  Recall of risky loans ("1") 94%
 
The summary: 

Logistic Regression model was applied in this analysis and the results showed high accuracy, precision, and recall. Therefore, logistic regression model is recommended for this machine learning analysis. To further enhance the precision of risky loans, larger sample size of historic data is required. 

