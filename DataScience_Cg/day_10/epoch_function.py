import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)

weights_input_hidden = np.random.uniform(size=(2,3))
bias_hidden = np.random.uniform(size=(1,3))

weights_hidden_output = np.random.uniform(size=(3,1))
bias_output = np.random.uniform(size=(1,1))

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):

    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    error = y - final_output

    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

hidden = sigmoid(np.dot(X, weights_input_hidden) + bias_hidden)
output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)

predictions = (output > 0.5).astype(int)

print("Final Output:")
print(output)

print("Predictions:")
print(predictions)

test_data = np.array([[3,4],[1,1]])
hidden_test = sigmoid(np.dot(test_data, weights_input_hidden) + bias_hidden)
final_test = sigmoid(np.dot(hidden_test, weights_hidden_output) + bias_output)

print("Test Output:")
print(final_test)