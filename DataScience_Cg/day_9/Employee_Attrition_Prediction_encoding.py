# 📖 Scenario: Employee Attrition Prediction 💼
# A company wants to predict whether an employee is likely to stay or leave based on their personal and workplace information. The dataset contains the following features:
# - age → Age of the employee (numeric)
# - years_experience → Number of years of work experience (numeric)
# - department → Department (HR, IT, Sales) (categorical)
# - education → Education level (Graduate, Postgraduate) (categorical)
# - attrition → Target variable (1 = Leaves, 0 = Stays)
# The data science team decides to build a machine learning pipeline that:
# - Standardizes numeric features (age, years_experience).
# - Converts categorical features (department, education) into numerical format using One-Hot Encoding.
# - Combines preprocessing and model training into a single pipeline.
# - Uses Logistic Regression to predict employee attrition.

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    'age': [25, 40, 30, 35, 28, 50],
    'years_experience': [2, 15, 5, 10, 3, 20],
    'department': ['HR', 'IT', 'Sales', 'IT', 'HR', 'Sales'],
    'education': ['Graduate', 'Postgraduate', 'Graduate', 'Postgraduate', 'Graduate', 'Postgraduate'],
    'attrition': [1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['age', 'years_experience', 'department', 'education']]
y = df['attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_features = ['age', 'years_experience']
categorical_features = ['department', 'education']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print("Model Accuracy:", accuracy)

new_employee = pd.DataFrame({
    'age': [32],
    'years_experience': [6],
    'department': ['IT'],
    'education': ['Graduate']
})

prediction = pipeline.predict(new_employee)

print("Prediction:", prediction)