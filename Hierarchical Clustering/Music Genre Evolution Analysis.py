import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("music_data.csv")



features = df[['Danceability', 'Energy', 'Tempo', 'Loudness']]

# -----------------------------
# Scaling
# -----------------------------

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

# -----------------------------
# Hierarchical Clustering
# -----------------------------

linkage_matrix = linkage(scaled_features, method='ward')

# -----------------------------
# Dendrogram
# -----------------------------

plt.figure(figsize=(12, 6))

dendrogram(linkage_matrix)

plt.title("Music Genre Relationship Mapping")
plt.xlabel("Songs")
plt.ylabel("Distance")

plt.show()