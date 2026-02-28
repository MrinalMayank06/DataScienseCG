# ==========================================
# Multiple Linear Regression
# Employee Salary Prediction
# ==========================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# 1️⃣ Create Dataset
data = {
    "Experience_years": [1,2,3,4,5,6,7,8,2,4,6,9],
    "Education_Level": [1,1,2,2,2,3,3,3,1,2,2,3],
    "Skills_Count": [3,4,5,6,7,8,9,10,3,5,7,10],
    "Performance_Rating": [3,3,4,4,5,5,4,5,2,3,4,5],
    "Salary_lpa": [4,5,7,8,10,12,13,15,4.5,7.5,11,16]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# 2️⃣ Define Features (X) and Target (y)
X = df[["Experience_years", "Education_Level", "Skills_Count", "Performance_Rating"]]
y = df["Salary_lpa"]


# 3️⃣ Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4️⃣ Train Model
model = LinearRegression()
model.fit(X_train, y_train)


# 5️⃣ Model Coefficients
print("\nModel Equation:")
print("Intercept (b0):", model.intercept_)
print("b1 (Experience):", model.coef_[0])
print("b2 (Education):", model.coef_[1])
print("b3 (Skills):", model.coef_[2])
print("b4 (Performance):", model.coef_[3])


# 6️⃣ Predict on Test Data
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")


# 7️⃣ Evaluate Model
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# 8️⃣ Predict Salary for New Employee
new_employee = pd.DataFrame({
    "Experience_years": [5],
    "Education_Level": [2],
    "Skills_Count": [7],
    "Performance_Rating": [4]
})

predicted_salary = model.predict(new_employee)

print("\nPredicted Salary for New Employee:", predicted_salary[0])