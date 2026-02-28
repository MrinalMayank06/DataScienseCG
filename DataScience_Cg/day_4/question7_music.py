import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"C:\Users\krish\Downloads\music_genre_dataset.csv")


print("Total Samples:", len(df))
print(df.head())

X = df.drop("genre_label", axis=1)
y = df["genre_label"]

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

print("\nDataset Split:")
print("Training Set:", len(X_train))
print("Validation Set:", len(X_val))
print("Test Set:", len(X_test))