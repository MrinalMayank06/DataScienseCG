from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

 
X = [[5,2,3],[4,1,4],[1,5,2],[2,4,1],[5,1,5],
     [3,5,1],[1,4,3],[5,3,4],[2,1,4],[3,4,2]]

y = [1, 1, 0, 0, 1, 0, 0, 1, 1, 0]

 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

 
for k in [1, 3, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    print(f"K={k}  Accuracy={acc:.2f}")

 
best_knn = KNeighborsClassifier(n_neighbors=3)
best_knn.fit(X_train, y_train)

 
new_user = [[4, 2, 4]]
new_user_scaled = scaler.transform(new_user)

prediction = best_knn.predict(new_user_scaled)[0]

print("Will they like it?", prediction)