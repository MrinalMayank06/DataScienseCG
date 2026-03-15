# Scenario Question: Predicting Titanic Survival
# Researchers are studying the Titanic disaster and want to build models that predict whether a
#  passenger would survive or not survive based on their information.
# - Features used:
# - Passenger class (pclass)
# - Gender (sex)
# - Age (age)
# - Number of siblings/spouses aboard (sibsp)
# - Number of parents/children aboard (parch)
# - Ticket fare (fare)
# - Label:
# - 1 = Survived
# - 0 = Died
# The researchers train three different models:
# - Logistic Regression
# - K-Nearest Neighbors (KNN) with k=5
# - Decision Tree with max depth = 4
# They then evaluate each model using a classification report (precision, recall, F1-score, accuracy).



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def load_data():
    df = sns.load_dataset("titanic")
    cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]
    return df[cols]


def preprocess(df):
    df = df.dropna()
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    return df


def split(df):
    X = df.drop("survived", axis=1)
    y = df["survived"]
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def train(X_train, y_train, X_train_scaled):
    models = {}

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    models["KNN"] = knn

    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt

    return models


def evaluate(models, X_test, X_test_scaled, y_test):
    results = {}

    for name, model in models.items():
        if name in ["Logistic Regression", "KNN"]:
            preds = model.predict(X_test_scaled)
        else:
            preds = model.predict(X_test)

        print(f"\n{name}")
        print(classification_report(y_test, preds))
        acc = accuracy_score(y_test, preds)
        print("Accuracy:", acc)
        results[name] = acc

    return results


def plot_results(results):
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.show()


df = load_data()
df = preprocess(df)

X_train, X_test, y_train, y_test = split(df)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = train(X_train, y_train, X_train_scaled)
results = evaluate(models, X_test, X_test_scaled, y_test)
plot_results(results)