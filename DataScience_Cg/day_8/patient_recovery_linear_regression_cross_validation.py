# Scenario: Predicting Patient Recovery Time
# A hospital research team wants to build a model to predict patient recovery time (in days) after
#  surgery based on several factors such as:
# - Age of the patient
# - Number of hours of post-surgery physiotherapy per week
# - Pre-existing health conditions (numeric severity score)
# - Length of hospital stay (days)
# - Average sleep hours during recovery
# They collect data from 1,000 patients and decide to use Linear Regression.
# To evaluate the model, they apply 5-Fold Cross-Validation with R² as the performance metric.


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

X, y = make_regression(n_samples=1000,
                       n_features=5,
                       noise=15,
                       random_state=42)

model = LinearRegression()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("R2 score for each fold:", scores.round(3))
print("Mean R2:", scores.mean().round(3))
print("Standard Deviation:", scores.std().round(3))

if scores.std() < 0.05:
    print("Model performance is stable across folds.")
else:
    print("Model performance varies across folds.")