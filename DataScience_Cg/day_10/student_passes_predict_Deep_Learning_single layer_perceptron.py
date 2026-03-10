# Scenario:
# Imagine we want to classify whether a student passes (1)
#  or fails (0) based on their study hours.
# - If study hours ≥ 5 → Pass
# - If study hours < 5 → Fail
# We’ll use a perceptron to learn this rule.

import numpy as np

# Step 1: Training Data
# X = study hours, y = pass/fail
X = np.array([[2], [4], [6], [8]])   # input: study hours
y = np.array([0, 0, 1, 1])           # output: fail=0, pass=1

# Step 2: Initialize weights and bias
weights = np.random.rand(1)   # single input → single weight
bias = np.random.rand(1)
learning_rate = 0.1

# Step 3: Activation Function (Step function)
def activation(z):
    return 1 if z >= 0 else 0


# Step 4: Training Loop
for epoch in range(10):  # run for 10 iterations
    print(f"Epoch {epoch+1}")
    for i in range(len(X)):
        # Linear combination
        z = np.dot(X[i], weights) + bias
        y_pred = activation(z)

        # Error
        error = y[i] - y_pred

        # Update rule
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

        print(f"Input: {X[i]}, Predicted: {y_pred}, Actual: {y[i]}, Error: {error}")

print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Step 5: Test the perceptron
test_hours = np.array([3, 7])
for h in test_hours:
    z = np.dot(h, weights) + bias
    print(f"Study Hours: {h}, Prediction: {activation(z)}")
