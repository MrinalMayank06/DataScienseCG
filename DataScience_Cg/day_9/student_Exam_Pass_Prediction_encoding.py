# Scenario: Student Exam Pass Prediction 🎓

# A university wants to predict whether a student will pass or fail an exam based
#  on their personal and academic information. The dataset contains the following features:

# age → Age of the student (numeric)

# hours_study → Number of hours the student studies per day (numeric)

# gender → Male or Female (categorical)

# school → School type (Government or Private) (categorical)

# pass_exam → Target variable (1 = Pass, 0 = Fail)

# The data science team decides to build a machine learning pipeline that:

# Standardizes numeric features (age, hours_study) so they are on the same scale.

# Converts categorical features (gender, school) into numerical format using One-Hot Encoding.

# Combines preprocessing and model training into a single pipeline.

# Uses Logistic Regression to predict whether a student will pass.

import pandas as pd

# Example dataset
data = {
    'age': [18, 19, 18, 20, 21],
    'hours_study': [2, 5, 1, 6, 4],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Male'],
    'school': ['Government', 'Private', 'Government', 'Private', 'Government'],
    'pass_exam': [0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Define features and target
X = df[['age', 'hours_study', 'gender', 'school']]
y = df['pass_exam']


# Import sklearn tools
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# Define preprocessing
numeric_features = ['age', 'hours_study']
categorical_features = ['gender', 'school']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Train the model
pipeline.fit(X, y)

# New student data
new_student = pd.DataFrame({
    'age': [19],
    'hours_study': [3],
    'gender': ['Female'],
    'school': ['Government']
})


# Prediction
prediction = pipeline.predict(new_student)

print("Prediction:", prediction)


 