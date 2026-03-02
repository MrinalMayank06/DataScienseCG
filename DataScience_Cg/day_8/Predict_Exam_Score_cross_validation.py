# Scenario: Predicting Student Exam Performance
# A university research team wants to build a model to predict student exam scores (out of 100) based on several factors such as:
# - Number of study hours per week
# - Attendance percentage in lectures
# - Prior GPA (Grade Point Average)
# - Participation in group projects (numeric engagement score)
# - Average sleep hours during exam preparation
# They collect data from 800 students across different departments and decide to use Linear Regression.
# To evaluate the model, they apply 5-Fold Cross-Validation with R² as the performance metric.

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

# Step 1: Create dataset (800 students, 5 features)
X, y = make_regression(n_samples=800,
                       n_features=5,
                       noise=12,
                       random_state=42)

# Step 2: Create Linear Regression model
model = LinearRegression()

# Step 3: Define 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Evaluate using R² metric
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# Step 5: Print results
print("R2 score for each fold:", scores.round(3))
print("Mean R2:", scores.mean().round(3))
print("Standard Deviation:", scores.std().round(3))

# Step 6: Interpretation
if scores.std() < 0.05:
    print("Model performance is stable across folds.")
else:
    print("Model performance varies across folds.")