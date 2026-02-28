import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\krish\Downloads\Delievery_dataset - Sheet1 (1).csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

X = df[["Distance_km",
        "Items",
        "Traffic_Level",
        "Processing_Time_hr"]]

y = df["Delivery_Time_hr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel Equation Coefficients:")
print("Intercept (b0):", model.intercept_)
print("b1 (Distance):", model.coef_[0])
print("b2 (Items):", model.coef_[1])
print("b3 (Traffic):", model.coef_[2])
print("b4 (Processing Time):", model.coef_[3])

y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_order = pd.DataFrame({
    "Distance_km": [15],
    "Items": [4],
    "Traffic_Level": [2],
    "Processing_Time_hr": [1.5]
})

predicted_time = model.predict(new_order)

print("\nPredicted Delivery Time (hours):", predicted_time[0])