import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = {
"Engine_Size": [1.2,1.5,1.8,2.0,2.2,1.3,1.6,2.4,2.0,1.4,1.7,2.5,1.8,2.2,1.5],
"Mileage": [90,70,60,50,40,85,65,30,45,80,55,25,50,35,75],
"Age": [8,6,5,4,3,7,6,2,4,7,5,1,3,2,6],
"Horsepower": [80,95,110,130,150,85,100,180,140,90,115,200,125,160,105],
"Price": [3.5,5,6,8,10,4,5.5,14,9,4.5,6.5,16,8.5,12,5.2]
}

df = pd.DataFrame(data)

print("First 5 Rows:")
print(df.head())

X = df[["Engine_Size", "Mileage", "Age", "Horsepower"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept (b0):", model.intercept_)
print("b1 (Engine Size):", model.coef_[0])
print("b2 (Mileage):", model.coef_[1])
print("b3 (Age):", model.coef_[2])
print("b4 (Horsepower):", model.coef_[3])

y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")

print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

new_car = pd.DataFrame({
    "Engine_Size": [2.0],
    "Mileage": [40],
    "Age": [3],
    "Horsepower": [150]
})

predicted_price = model.predict(new_car)

print("\nPredicted Car Price:", predicted_price[0])