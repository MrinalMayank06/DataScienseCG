#import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

#create dataset
data = {
    "area shift": [600, 800, 1000, 1200, 1400, 1600],
    "price lakhs": [40, 50, 60, 70, 80, 90]  # Added missing values
}
df = pd.DataFrame(data)
print("dataset: ")
print(df)

X = df[["area shift"]]  # Capital X for consistency
y = df["price lakhs"]    # Fixed column name (removed underscore)

# Fixed train_test_split syntax and variable names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)  # Fixed: was Y_test instead of y_train

print("\nslope(m):", model.coef_[0])
print("intercept(b):", model.intercept_)

y_pred = model.predict(X_test)  # Fixed: was x_test (should be X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):  # Fixed: was Y_test
    print(f"Actual:{actual:.2f}, predicted:{pred:.2f}")

#evaluate model
mae = mean_absolute_error(y_test, y_pred)  # Fixed: was Y_test
r2 = r2_score(y_test, y_pred)  # Fixed: was Y_test

print("\nMean Absolute error: ", mae)  # Fixed: was /n
print("R2 score:", r2)

new_area = np.array([[1800]])
predicted_price = model.predict(new_area)
print(f"\nPredicted price for 1800 sqft house: {predicted_price[0]:.2f} lakhs")

#VISUALISATION
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")  # Fixed: was model.predict(x)
plt.xlabel("Area (sqft)")
plt.ylabel("Price (lakhs)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()
plt.show()