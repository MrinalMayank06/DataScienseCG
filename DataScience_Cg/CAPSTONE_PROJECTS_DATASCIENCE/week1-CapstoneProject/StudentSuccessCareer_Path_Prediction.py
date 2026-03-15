# Capstone Project: Student Success & Career Path Prediction
# Scenario
# The university wants to analyze student performance data to:
# Predict exam scores (Regression).
# Classify students into “At Risk” vs. “On Track” categories (Classification).
# Cluster students into groups with similar study habits (Clustering).
# Recommend interventions (extra tutoring, workshops, counseling).
# https://github.com/himanshusar123/Datasets
# Student Success and Career Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_squared_error, r2_score,
    classification_report, accuracy_score,
    confusion_matrix
)

df = pd.read_csv(r"c:\Users\krish\Downloads\Student Success & Career Path  - Sheet1.csv")

df = df.dropna()

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Pass_Fail"] = le.fit_transform(df["Pass_Fail"])

df = df.drop("Student_ID", axis=1)

X_reg = df.drop("Final_Exam_Score", axis=1)
y_reg = df["Final_Exam_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

reg_model = LinearRegression()
reg_model.fit(X_train_scaled, y_train)

y_pred_reg = reg_model.predict(X_test_scaled)

print("Regression Results")
print("R2 Score:", r2_score(y_test, y_pred_reg))
print("MSE:", mean_squared_error(y_test, y_pred_reg))

plt.figure()
plt.scatter(y_test, y_pred_reg)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.show()

df["status"] = np.where(df["Final_Exam_Score"] < 50, 0, 1)

X_clf = df.drop(["Final_Exam_Score", "status"], axis=1)
y_clf = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("\nLogistic Regression")
print(classification_report(y_test, y_pred_log))
print("Accuracy:", accuracy_score(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
plt.figure()
sns.heatmap(cm_log, annot=True, fmt="d")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\nKNN Results")
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))

cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure()
sns.heatmap(cm_knn, annot=True, fmt="d")
plt.title("KNN Confusion Matrix")
plt.show()

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Results")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure()
sns.heatmap(cm_dt, annot=True, fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.show()

cluster_features = df[[
    "Hours_Studied",
    "Attendance (%)",
    "Assignments_Submitted",
    "Participation_Score"
]]

cluster_scaled = scaler.fit_transform(cluster_features)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(cluster_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(cluster_scaled)

print("\nCluster Distribution")
print(df["cluster"].value_counts())

def recommend(row):
    if row["status"] == 0 and row["Attendance (%)"] < 60:
        return "Counseling + Attendance Monitoring"
    elif row["status"] == 0:
        return "Extra Tutoring"
    elif row["cluster"] == 2:
        return "Advanced Workshops"
    else:
        return "Regular Monitoring"

df["recommendation"] = df.apply(recommend, axis=1)

print("\nSample Recommendations")
print(df[["Final_Exam_Score", "status", "cluster", "recommendation"]].head())