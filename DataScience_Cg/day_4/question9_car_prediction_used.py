import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\krish\Downloads\used_car_condition_dataset.csv")

print("Full Dataset:")
print(df)

print("\nFirst 5 Rows:")
print(df.head())

print("\nTotal Samples:", len(df))

X = df.drop("condition_label", axis=1)
y = df["condition_label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining Set:", len(X_train))
print("Validation Set:", len(X_val))
print("Test Set:", len(X_test))