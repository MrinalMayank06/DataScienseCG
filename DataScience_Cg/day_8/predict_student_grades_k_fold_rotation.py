# Scenario-Based Question
# A university wants to build a predictive model to estimate
#  student grades based on four factors:
# - Study hours per week
# - Attendance percentage
# - Previous exam score
# - Average sleep hours
# They collect data from 200 students and decide to use Ridge
#  Regression for prediction. To evaluate the model, they apply different cross-validation strategies:
# - Basic K-Fold CV (5 folds, shuffled) to check the stability of the model’s R² scores.
# - Multi-metric evaluation using both R² and Mean Squared Error (MSE), comparing training and validation
# scores.
# - Stratified K-Fold CV (for a separate classification task predicting pass/fail using Logistic
#                         Regression).




from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

# Step 1: Create dataset (200 students, 4 features)
X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)

# Step 2: Create model
model = LinearRegression()

# Step 3: Define K-Fold strategy
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Step 4: Evaluate model using R²
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

# Step 5: Print results
print("R2 scores for each fold:", scores.round(3))
print("Mean R2:", scores.mean().round(3))
print("Std Dev:", scores.std().round(3))

# Step 6: Interpret results
if scores.std() < 0.05:
    print("Model is stable across folds.")
else:
    print("Model performance varies across folds, investigate further.")