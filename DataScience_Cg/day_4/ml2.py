import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset (change file name if different)
df = pd.read_csv("music_dataset.csv")

# Assume:
# 'audio_path' → file name / features
# 'genre' → target label

X = df.drop("genre", axis=1)   # features
y = df["genre"]                # target

# 70% Train, 30% Temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Split remaining 30% into 15% Val, 15% Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))