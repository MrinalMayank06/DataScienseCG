# Scenario-Based Question: Customer Purchase Prediction using Pipeline and GridSearchCV 🛒

# A retail company wants to predict whether a customer will purchase a product (Yes/No) based on
# their profile. The dataset contains the following features:

# age → Age of the customer

# income → Monthly income

# gender → Male/Female

# city → City of residence

# purchased → Target variable (1 = Purchased, 0 = Not Purchased)

# The data science team has already created a Pipeline that performs:

# Scaling of numeric features using StandardScaler

# Encoding of categorical features using OneHotEncoder

# Classification using LogisticRegression




# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# 1. Load dataset (example)
# -----------------------------
# Assume df is your DataFrame with columns: age, income, gender, city, purchased
# df = pd.read_csv("customer_data.csv")

# Creating a sample DataFrame for demonstration purposes
data = {
    'age': [25, 30, 35, 40, 22, 28, 45, 50, 32, 38],
    'income': [30000, 50000, 70000, 60000, 35000, 48000, 90000, 80000, 55000, 65000],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Male'],
    'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'New York', 'Chicago', 'Los Angeles', 'Houston', 'New York', 'Chicago'],
    'purchased': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# Features and target
X = df.drop("purchased", axis=1)
y = df["purchased"]

# -----------------------------
# 2. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 3. Define preprocessing
# -----------------------------
numeric_features = ["age", "income"]
categorical_features = ["gender", "city"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# -----------------------------
# 4. Build pipeline
# -----------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# -----------------------------
# 5. Define parameter grid for GridSearchCV
# -----------------------------
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__penalty": ["l2"],  # l1 requires solver='liblinear'
    "classifier__solver": ["lbfgs", "saga"]
}

# -----------------------------
# 6. Apply GridSearchCV
# -----------------------------
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3, # Changed from 5 to 3 to accommodate the minority class size
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# -----------------------------
# 7. Evaluate best model
# -----------------------------
print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
