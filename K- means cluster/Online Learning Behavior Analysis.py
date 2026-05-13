# Online Learning Behavior Analysis

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dataset
data = {
    'WatchTime': [2, 10, 3, 15, 5, 12, 4, 18],
    'QuizPerformance': [40, 95, 50, 98, 60, 90, 55, 99],
    'LoginFrequency': [1, 8, 2, 10, 4, 9, 3, 11]
}

df = pd.DataFrame(data)

# Features
X = df[['WatchTime', 'QuizPerformance', 'LoginFrequency']]

# K-Means
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(X)

# Output
print(df)

# Student Engagement Clusters
print("\nStudent Engagement Clusters:")
print(df[['WatchTime', 'QuizPerformance', 'Cluster']])

# Visualization
plt.scatter(df['WatchTime'], df['QuizPerformance'], c=df['Cluster'])
plt.xlabel("Video Watch Time")
plt.ylabel("Quiz Performance")
plt.title("Student Learning Clusters")
plt.show()