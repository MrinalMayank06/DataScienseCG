# 🎬 Scenario: Movie Streaming Platform
# A movie streaming company has collected data on 1,000 users, including:
# - Average watch time per week
# - Preferred genres (action, comedy, drama, etc.)
# - Number of devices used (TV, phone, tablet)
# - Frequency of subscription pauses or cancellations
# The company wants to group users into meaningful segments to:
# - Recommend personalized movie lists
# - Design loyalty rewards for binge‑watchers
# - Identify users at risk of canceling subscriptions
# Your Tasks
# - Apply K‑Means clustering to explore possible user segments.
# - Example clusters: “Weekend binge‑watchers,” “Casual family viewers,” “Genre loyalists.”
# - Use the Elbow Method to find the point where adding more clusters doesn’t improve grouping much.
# - This helps decide whether 3, 4, or 5 clusters make sense.
# - Validate with Silhouette Score to check if the chosen clusters are well‑separated and meaningful.
# - Ensures that “binge‑watchers” aren’t mixed up with “casual viewers.”



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

file_path = r"C:\Users\krish\Downloads\movie_streaming_users_1000.csv"
df = pd.read_csv(file_path)

label_encoder = LabelEncoder()
df["Genre_Encoded"] = label_encoder.fit_transform(df["Preferred_Genre"])

features = df[
    [
        "Avg_Watch_Time_per_Week_Hours",
        "Genre_Encoded",
        "Number_of_Devices_Used",
        "Subscription_Pause_Frequency_per_Year",
    ]
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia_values = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, inertia_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

silhouette_scores = []
k_range_sil = range(2, 11)

for k in k_range_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    silhouette_scores.append(score)

plt.figure()
plt.plot(k_range_sil, silhouette_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis")
plt.show()

optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans_final.fit_predict(scaled_features)

plt.figure()
plt.scatter(
    df["Avg_Watch_Time_per_Week_Hours"],
    df["Subscription_Pause_Frequency_per_Year"],
    c=df["Cluster"]
)
plt.xlabel("Avg Watch Time per Week")
plt.ylabel("Pause Frequency per Year")
plt.title("Cluster Visualization (k=3)")
plt.show()

final_score = silhouette_score(scaled_features, df["Cluster"])
print("Final Silhouette Score:", final_score)