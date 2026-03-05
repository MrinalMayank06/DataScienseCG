# Scenario:
# We want to classify whether a customer buys (1) or does not buy (0) a product based on two features:
# - Ad Clicks (how many times they clicked an online ad)
# - Time on Website (minutes spent browsing)
# This is a slightly more complex problem, so we’ll use a multi-layer neural network with one hidden layer.

import numpy as np

# Step 1: Training Data
# X = [ad_clicks, time_on_site], y = buy(1)/not buy(0)
X = np.array([[1, 2],   # few clicks, short time → no buy
              [2, 1],   # few clicks, short time → no buy
              [4, 5],   # more clicks, longer time → buy
              [5, 6]])  # more clicks, longer time → buy
y = np.array([[0], [0], [1], [1]])

# Step 2: Initialize weights and biases
np.random.seed(42)
weights_input_hidden = np.random.rand(2, 2)   # 2 inputs → 2 hidden neurons
bias_hidden = np.random.rand(1, 2)

weights_hidden_output = np.random.rand(2, 1)  # 2 hidden → 1 output neuron
bias_output = np.random.rand(1, 1)

learning_rate = 0.1

# Step 3: Activation Function (Sigmoid for smooth output)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

# Step 4: Training Loop
for epoch in range(1000):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Error
    error = y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Step 5: Test the network
test_data = np.array([[3, 4], [1, 1]])
hidden_test = sigmoid(np.dot(test_data, weights_input_hidden) + bias_hidden)
final_test = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)

print("Predictions for test data:", final_test)
