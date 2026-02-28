import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler

# Employee Data (Age, Annual Salary)
data = np.array([
    [25, 15000],
    [28, 16000],
    [30, 18000],
    [35, 22000],
    [40, 25000],
    [45, 60000],
    [50, 65000],
    [55, 70000]
])

# Step 1: Scale Data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 2: Perform Hierarchical Clustering (Ward)
Z = linkage(data_scaled, method='ward')

# Step 3: Plot Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z, labels=[f"Emp {i}" for i in range(len(data))])
plt.title("Employee Segmentation Dendrogram")
plt.xlabel("Employees")
plt.ylabel("Ward Distance")
plt.axhline(y=1.5, linestyle='--')
plt.show()

# Step 4: Form 3 Clusters
clusters = fcluster(Z, 3, criterion='maxclust')

print("Cluster Labels:", clusters)

# Step 5: Visualize Final Clusters
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=clusters)
plt.xlabel("Age")
plt.ylabel("Annual Salary")
plt.title("Employee Segments")
plt.show()