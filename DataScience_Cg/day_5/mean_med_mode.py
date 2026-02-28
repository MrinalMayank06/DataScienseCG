# ==============================
# Multiple Linear Regression
# House Price Prediction Model
# ==============================

# 1️⃣ Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# 2️⃣ Create Dataset
data = {
    "Area_sqft": [800,1000,1200,1500,1800,2000,2200,2500,900,1600,1400,2100],
    "Bedrooms": [2,2,3,3,4,4,4,5,2,3,3,4],
    "Age_years": [15,10,8,5,4,3,2,1,12,6,7,3],
    "Distance_km": [12,10,8,6,5,4,3,2,11,7,9,4],
    "Price_lakhs": [40,50,62,75,90,105,120,140,45,80,70,110]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)


# 3️⃣ Define Features (Independent Variables) and Target (Dependent Variable)
X = df[["Area_sqft","Bedrooms","Age_years","Distance_km"]]
y = df["Price_lakhs"]


# 4️⃣ Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5️⃣ Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)


# 6️⃣ Predict on Test Data
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")


# 7️⃣ Evaluate Model Performance
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# 8️⃣ Predict Price for a New House
new_house = pd.DataFrame({
    "Area_sqft": [1700],
    "Bedrooms": [3],
    "Age_years": [5],
    "Distance_km": [6]
})

predicted_price = model.predict(new_house)
print("\nPredicted Price for New House:", predicted_price[0])