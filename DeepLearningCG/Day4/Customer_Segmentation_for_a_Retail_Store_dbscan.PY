#   M1 Capgemini: Scenario: Customer Segmentation for a Retail Store
# Context:
# A retail store wants to understand its customers better. They have data on how much each customer spends per visit and how frequently they visit the store. The goal is to group customers into clusters (e.g., “loyal high spenders,” “occasional visitors,” etc.) using DBSCAN, which is great for finding clusters of varying shapes and identifying outliers.
#   M1 Capgemini: data = np.array([
#     [5000, 50], [5200, 48], [4800, 52],   # Loyal high spenders
#     [1500, 10], [1600, 12], [1400, 9],    # Occasional visitors
#     [3000, 25], [3100, 27], [2900, 24],   # Moderate spenders
#     [8000, 5],                            # Outlier: rare but huge spender
#     [200, 2],                             # Outlier: very low engagement
# ])


# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler

# # Customer data: [Amount Spent, Visit Frequency]
# data = np.array([
#     [5000, 50], [5200, 48], [4800, 52],   # Loyal high spenders
#     [1500, 10], [1600, 12], [1400, 9],    # Occasional visitors
#     [3000, 25], [3100, 27], [2900, 24],   # Moderate spenders
#     [8000, 5],                            # Outlier
#     [200, 2],                             # Outlier
# ])

# # Step 1: Standardize data
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(data)

# # Step 2: Apply DBSCAN
# dbscan = DBSCAN(eps=0.8, min_samples=2)
# clusters = dbscan.fit_predict(scaled_data)

# # Step 3: Print cluster labels
# print("Cluster Labels:")
# print(clusters)

# # Step 4: Visualization
# plt.scatter(data[:,0], data[:,1], c=clusters, cmap='rainbow', s=100)
# plt.xlabel("Amount Spent per Visit")
# plt.ylabel("Visit Frequency")
# plt.title("Customer Segmentation using DBSCAN")
# plt.show()




# Customer Segmentation using DBSCAN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# Step 1: Create Customer Dataset
# [Amount Spent per Visit, Visit Frequency]

data = np.array([
    [5000, 50], [5200, 48], [4800, 52],   # Loyal high spenders
    [1500, 10], [1600, 12], [1400, 9],    # Occasional visitors
    [3000, 25], [3100, 27], [2900, 24],   # Moderate spenders
    [8000, 5],                            # Outlier: rare but huge spender
    [200, 2],                             # Outlier: very low engagement
])


# Step 2: Standardize the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# Step 3: Apply DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=2)
clusters = dbscan.fit_predict(scaled_data)


# Step 4: Print Cluster Labels
print("Cluster Labels:")
print(clusters)


# Step 5: Show Each Customer with Cluster
print("\nCustomer Segmentation Result:\n")
for i, point in enumerate(data):
    print(f"Customer {i+1} -> Spent: {point[0]}, Visits: {point[1]}, Cluster: {clusters[i]}")


# Step 6: Visualization
plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], c=clusters, cmap='rainbow', s=120)

plt.xlabel("Amount Spent per Visit")
plt.ylabel("Visit Frequency")
plt.title("Customer Segmentation using DBSCAN")

plt.grid(True)
plt.show()

