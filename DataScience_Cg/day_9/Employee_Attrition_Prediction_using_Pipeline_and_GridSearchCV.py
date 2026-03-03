# Scenario: Employee Attrition Prediction using Pipeline and GridSearchCV 👨‍💼

# A company wants to predict whether an employee will leave the company (Attrition) or stay, so HR can take preventive action. The dataset contains the following features:

# age → Age of the employee

# years_experience → Total years of work experience

# department → Department (Sales, IT, HR)

# education_level → Bachelor, Master, PhD

# attrition → Target variable (1 = Employee leaves, 0 = Employee stays)

# The data science team builds a Pipeline that:

# Scales numeric features using StandardScaler

# Encodes categorical features using OneHotEncoder

# Uses Logistic Regression for classification

# They use GridSearchCV to find the best hyperparameters:

# C: 0.1, 1, 10

# solver: liblinear, lbfgs

# They evaluate performance using 5-fold cross-validation and accuracy.


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = {
    'age': [25, 40, 30, 35, 28, 50, 45, 32, 29, 38],
    'years_experience': [2, 15, 5, 10, 3, 20, 18, 7, 4, 12],
    'department': ['HR', 'IT', 'Sales', 'IT', 'HR', 'Sales', 'IT', 'Sales', 'HR', 'IT'],
    'education_level': ['Bachelor', 'Master', 'Bachelor', 'PhD', 'Master', 'PhD', 'Master', 'Bachelor', 'Bachelor', 'PhD'],
    'attrition': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop('attrition', axis=1)
y = df['attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_features = ['age', 'years_experience']
categorical_features = ['department', 'education_level']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

y_pred = grid_search.predict(X_test)

print(classification_report(y_test, y_pred))