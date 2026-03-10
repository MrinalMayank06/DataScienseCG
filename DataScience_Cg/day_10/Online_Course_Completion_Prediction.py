# Scenario: Online Course Completion Prediction
# 
# Objective:
# Predict whether a student will complete an online course (1) or drop out (0)
# based on their engagement metrics.
#
# Features:
# 1. Videos Watched – Number of course videos the student watched
# 2. Time Spent on Platform – Total minutes the student spent on the learning platform
#
# Intuition:
# Students who watch more videos and spend more time on the platform
# are more likely to complete the course. However, the relationship 
# between these engagement features and course completion is not strictly linear.
#
# Training Data Example:
# Videos Watched | Time on Platform (min) | Complete Course
# 2              | 15                     | 0  (dropout)
# 3              | 20                     | 0  (dropout)
# 8              | 60                     | 1  (completed)
# 9              | 75                     | 1  (completed)
#
# Modeling Approach:
# - Use a Multi-Layer Neural Network (MLP) with one hidden layer.
# - The hidden layer captures non-linear interactions between features,
#   such as a combination of "high videos watched" AND "high time spent".
# - Output layer uses a sigmoid activation to predict probability of course completion.
# - This setup allows the network to learn complex patterns beyond simple linear rules.

import numpy as np

# Data
X = np.array([[2,15],
              [3,20],
              [8,60],
              [9,75]])

y = np.array([[0],[0],[1],[1]])

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights
np.random.seed(1)
W1 = np.random.randn(2,3)
b1 = np.zeros((1,3))

W2 = np.random.randn(3,1)
b2 = np.zeros((1,1))

learning_rate = 0.01

# Training
for epoch in range(5000):

    # Forward
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Loss derivative
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Test
test = np.array([[5,40]])
hidden = relu(np.dot(test, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)

print("Completion Probability:", output)