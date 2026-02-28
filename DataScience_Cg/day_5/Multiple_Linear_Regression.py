# ==========================================
# Multiple Linear Regression
# Employee Salary Prediction Model
# ==========================================

# 1️⃣ Import Required Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# 2️⃣ Create Sample Dataset
data = {
    "Experience_years": [1,2,3,4,5,6,7,8,9,10],
    "Education_Level": [1,1,2,2,2,3,3,3,2,3],   
    "Skills_Count": [3,4,5,6,6,7,8,9,7,10],
    "Performance_Rating": [3,3,4,4,5,4,5,5,4,5],
    "Salary_lpa": [3,4,6,8,10,12,15,18,16,20]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# 3️⃣ Define Features (X) and Target (y)
X = df[["Experience_years", "Education_Level", "Skills_Count", "Performance_Rating"]]
y = df["Salary_lpa"]


# 4️⃣ Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5️⃣ Train the Model
model = LinearRegression()
model.fit(X_train, y_train)


# 6️⃣ Model Equation Coefficients
print("\nModel Coefficients:")
print("Intercept (b0):", model.intercept_)
print("b1 (Experience):", model.coef_[0])
print("b2 (Education):", model.coef_[1])
print("b3 (Skills):", model.coef_[2])
print("b4 (Performance):", model.coef_[3])


# 7️⃣ Predict on Test Data
y_pred = model.predict(X_test)

print("\nActual vs Predicted:")
for actual, pred in zip(y_test, y_pred):
    print(f"Actual: {actual:.2f}, Predicted: {pred:.2f}")


# 8️⃣ Evaluate Model
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# 9️⃣ Predict Salary for New Employee
new_employee = pd.DataFrame({
    "Experience_years": [5],
    "Education_Level": [2],
    "Skills_Count": [7],
    "Performance_Rating": [4]
})

predicted_salary = model.predict(new_employee)

print("\nPredicted Salary for New Employee:", predicted_salary[0])