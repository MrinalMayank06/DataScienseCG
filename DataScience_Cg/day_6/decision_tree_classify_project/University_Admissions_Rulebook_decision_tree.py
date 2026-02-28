import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r"C:\Users\krish\Downloads\University Dataset - Sheet1.csv")

X = df[["HighSchool_GPA", "Exam_Score", "Extracurriculars"]]
y = df["Admission_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

model = DecisionTreeClassifier(criterion="gini", max_depth=3)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

new_student = [[8.2, 85, 1]]
print("Admission Decision:", model.predict(new_student)[0])