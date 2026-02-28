# # Scenario Question 💼
# # A telecommunications company has collected data on 500 customers, including their monthly bill amount, average call duration, internet usage, and number of support calls. The company wants to group customers into meaningful segments to design targeted marketing campaigns and improve customer service.
# # You are tasked with:
# # - Using K‑Means clustering to explore possible customer segments.
# # - Applying the Elbow Method to determine where adding more clusters stops giving significant improvement.
# # - Using the Silhouette Score to validate which number of clusters produces the most well‑separated and meaningful groups
  # Step 2: Create synthetic telecom customer dataset
# # (in practice, replace with actual enterprise customer data)
# data = {
#     'CustomerID': range(1, 501),
#     'MonthlyBill': np.random.randint(20, 200, 500),       # monthly bill in $
#     'CallDuration': np.random.randint(50, 500, 500),      # avg monthly call minutes
#     'InternetUsage': np.random.randint(10, 300, 500),     # GB per month
#     'SupportCalls': np.random.randint(0, 10, 500)         # number of support calls
# }
# df = pd.DataFrame(data)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Create Synthetic Dataset
data = {
    'CustomerID': range(1, 501),
    'MonthlyBill': np.random.randint(20, 200, 500),
    'CallDuration': np.random.randint(50, 500, 500),
    'InternetUsage': np.random.randint(10, 300, 500),
    'SupportCalls': np.random.randint(0, 10, 500)
}

df = pd.DataFrame(data)

# Step 2: Select Features
X = df[['MonthlyBill', 'CallDuration', 'InternetUsage', 'SupportCalls']]

# Step 3: Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Inertia)")
plt.title("Elbow Method")
plt.show()

# Step 5: Silhouette Score Plot
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

plt.figure()
plt.plot(k_range, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

# Step 6: Final Model (Assume best K from analysis)
optimal_k = 3  # Change based on elbow + silhouette
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)

print(df.head())