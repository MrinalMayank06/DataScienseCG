# Scenario:
# We want to classify whether a restaurant order is “Large” (1) or “Small” (0) based on the number
#  of items ordered.
# - If items ≥ 3 → Large order
# - If items < 3 → Small order




import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

np.random.seed(42)
weight = np.random.rand()
bias = np.random.rand()

learning_rate = 0.1

def activation(z):
    return 1 if z >= 0 else 0

epochs = 20

for epoch in range(epochs):
    for i in range(len(X)):
        z = np.dot(X[i], weight) + bias
        y_pred = activation(z)
        error = y[i] - y_pred
        weight += learning_rate * error * X[i][0]
        bias += learning_rate * error

print("Training Complete")
print("Final Weight:", weight)
print("Final Bias:", bias)

test_items = np.array([1, 2, 3, 4, 6])

print("\nTesting Predictions:")
for items in test_items:
    z = items * weight + bias
    prediction = activation(z)
    order_type = "Large" if prediction == 1 else "Small"
    print(f"Items: {items} → {order_type}")