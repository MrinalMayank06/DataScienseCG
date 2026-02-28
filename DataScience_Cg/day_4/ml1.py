# Scenario: AI Model for Detecting Defective Products in a Factory
# A manufacturing company wants to build an AI system that detects defective products from camera images on the production line.
# They have collected 100 product images, and each image is labeled as:
# 0 → Non-defective product
# 1 → Defective product
# To build a reliable AI model, the data scientist must divide the dataset into three parts:
# Training set (70%) → Used to train the AI model
# Validation set (15%) → Used to tune and improve the model
# Test set (15%) → Used to evaluate final performance
# Your task is to help the data scientist split the dataset correctly.

import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([f"product_{i}.jpg" for i in range(100)])
y = np.array([0]*80 + [1]*20)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}")
print(f"Val:   {len(X_val)}")
print(f"Test:  {len(X_test)}")