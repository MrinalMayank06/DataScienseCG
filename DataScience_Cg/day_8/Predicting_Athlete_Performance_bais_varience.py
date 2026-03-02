# Scenario: Predicting Athlete Performance
# A sports academy wants to build a model to predict athlete sprint times (in seconds) based on training
#  hours. They collect data from 30 athletes, but the sprint times are noisy because of other factors (like diet, fatigue, or weather).
# They try two different models:
# - Linear Model (straight line) → very simple, assumes sprint times improve perfectly with more
#  training hours.
# - Polynomial Model (degree 10 curve) → very complex, tries to follow every bump in the data.

# Questions
# - Part A: If the linear model consistently predicts sprint times that are too fast or too slow compared
#  to actual results, what does this show about bias?
# - Part B: If the polynomial model fits the training data almost perfectly but gives very different
# predictions when tested on new athletes, what does this show about variance?
# - Part C: Which model is likely to generalize better to new athletes, and why?
# - Part D (Applied): How would you explain the difference between “high bias” and “high variance”
#  to a coach who doesn’t know machine learning?



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Step 1: Generate Data
np.random.seed(0)

X = np.linspace(0, 10, 30).reshape(-1, 1)   # training hours

# Sprint time decreases with training but has nonlinear pattern + noise
y = (20 - 2 * X + 0.3 * X**2).ravel() + np.random.normal(scale=2, size=30)

# Step 2: Define models
linear_model = make_pipeline(PolynomialFeatures(1), LinearRegression())   # simple line
poly_model = make_pipeline(PolynomialFeatures(10), LinearRegression())    # complex curve
balanced_model = make_pipeline(PolynomialFeatures(3), LinearRegression()) # balanced

# Step 3: Fit models
linear_model.fit(X, y)
poly_model.fit(X, y)
balanced_model.fit(X, y)

# Step 4: Predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)

y_linear = linear_model.predict(X_test)
y_poly = poly_model.predict(X_test)
y_balanced = balanced_model.predict(X_test)

# Step 5: Plot
plt.figure(figsize=(12, 4))

# High Bias
plt.subplot(1, 3, 1)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_linear, color="red", label="Linear Model")
plt.title("High Bias (Underfitting)")
plt.legend()

# Balanced
plt.subplot(1, 3, 2)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_balanced, color="green", label="Poly deg=3")
plt.title("Balanced Model")
plt.legend()

# High Variance
plt.subplot(1, 3, 3)
plt.scatter(X, y, color="gray", label="Data")
plt.plot(X_test, y_poly, color="blue", label="Poly deg=10")
plt.title("High Variance (Overfitting)")
plt.legend()

plt.tight_layout()
plt.show() 