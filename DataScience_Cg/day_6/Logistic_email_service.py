import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
X = np.array([
    [50, 1, 0.8],
    [200, 0, 0.1],
    [30, 1, 0.9],
    [180, 0, 0.05],
    [10, 1, 0.95],
    [220, 0, 0.08],
])

y = np.array([1, 0, 1, 0, 1, 0])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

# Accuracy
print("acc:",y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("probab",proba.round(2))


#new mail 
new_e= [[89,2,0.99]]
print(" new email is spam ? ",model.predict(new_e)[0])



