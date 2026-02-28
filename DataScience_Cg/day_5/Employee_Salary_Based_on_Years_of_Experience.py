import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(r"C:\Users\krish\Downloads\salary_lpa - Sheet1.csv")

print(df.head())
print(df.columns)

X = df[['Experience_years']]
y = df['Salary_lpa']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("Model Equation:")
print(f"Salary_lpa = {model.coef_[0]:.2f} * Experience_years + {model.intercept_:.2f}")

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Experience_years")
plt.ylabel("Salary_lpa")
plt.title("Experience vs Salary")
plt.show()

experience_input = float(input("Enter Years of Experience: "))
predicted_salary = model.predict([[experience_input]])

print(f"Predicted Salary: {predicted_salary[0]:.2f} LPA")