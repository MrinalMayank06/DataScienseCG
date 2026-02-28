#   # Hierarchircal Clustering

# # Scenario Question 💼
# # A retail bank wants to understand its customers better. They have
# # collected data on Age and Annual Income for a sample of customers.
# #  The goal is to group customers into meaningful segments so the bank can
# #  design targeted loan offers, personalized investment plans, and marketing campaigns.
#  : data = np.array([
#     [25, 15000],
#     [28, 16000],
#     [30, 18000],
#     [35, 22000],
#     [40, 25000],
#     [45, 60000],
#     [50, 65000],
#     [55, 70000]
# ])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

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

linked = linkage(data, method='ward')

plt.figure()
dendrogram(linked)
plt.title("Customer Dendrogram age vs income3")
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.show()

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = model.fit_predict(data)

plt.figure()
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.xlabel("Age")
plt.ylabel("Annual Income")
plt.title("Hierarchical Clustering Result")
plt.show()

print("Cluster Labels:", labels)

