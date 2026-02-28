# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

 
df = pd.read_csv(r"C:\Users\krish\Downloads\Fitness_app_dataset - Sheet1.csv")

print("Dataset Preview:")
print(df.head())

 
X = df[['Exercise', 'Diet', 'Stress']]
y = df['AtRisk']

 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

 
k_values = [1, 3, 5]
best_k = None
best_accuracy = 0

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"K = {k}, Accuracy = {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\nBest K = {best_k} with Accuracy = {best_accuracy:.4f}")

 
best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train_scaled, y_train)

 
new_user = np.array([[4, 3, 4]])
new_user_scaled = scaler.transform(new_user)

prediction = best_model.predict(new_user_scaled)

if prediction[0] == 1:
    print("Prediction: User is AT RISK of heart disease.")
else:
    print("Prediction: User is NOT at risk of heart disease.")