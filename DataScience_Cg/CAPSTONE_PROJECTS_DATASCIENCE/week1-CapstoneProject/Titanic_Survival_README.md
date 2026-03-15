# Predicting Titanic Survival

This project explores how machine learning models can be used to predict
whether a passenger aboard the Titanic would survive or not.

The project trains multiple classification algorithms and compares their
performance using evaluation metrics.

------------------------------------------------------------------------

# Problem Description

Researchers are analyzing passenger data from the Titanic disaster to
determine which factors influenced survival.

The model attempts to predict whether a passenger survived based on the
following features:

Passenger Class (pclass)

Gender (sex)

Age

Number of Siblings/Spouses aboard (sibsp)

Number of Parents/Children aboard (parch)

Ticket Fare

Target label:

1 = Survived

0 = Did Not Survive

------------------------------------------------------------------------

# Machine Learning Models Used

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Each model is trained and evaluated to determine which performs best.

------------------------------------------------------------------------

# Workflow

Load dataset

Preprocess data

Split data into training and testing sets

Scale numerical features

Train machine learning models

Evaluate models

Compare accuracy

Visualize results

------------------------------------------------------------------------

# How to Run the Project

Step 1: Install required libraries

pip install pandas seaborn matplotlib scikit-learn

Step 2: Run the script

python Predicting_Titanic_Survival.py

The program will train three models and display their performance
metrics.

------------------------------------------------------------------------

# Model Evaluation

Each model is evaluated using:

Precision

Recall

F1-score

Accuracy

The results are printed using a classification report.

------------------------------------------------------------------------

# Visualization

A bar chart is generated to compare the accuracy of each model.

This helps quickly identify which model performs best for the Titanic
survival prediction task.

------------------------------------------------------------------------

# Learning Outcomes

This project demonstrates:

Binary classification techniques

Feature preprocessing

Model evaluation

Comparison of machine learning algorithms
